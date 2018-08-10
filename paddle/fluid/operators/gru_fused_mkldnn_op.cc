/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using std::vector;
using framework::Tensor;
using platform::MKLDNNDeviceContext;

using paddle::framework::Tensor;
using paddle::framework::LoDTensor;
using paddle::framework::LoD;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::CPUDeviceContext;
using paddle::platform::MKLDNNMemDesc;
using paddle::platform::MKLDNNGetDataType;
using framework::DataLayout;
using framework::vectorize2int;
using mkldnn::memory;
using mkldnn::rnn_cell;
using mkldnn::rnn_forward;
using mkldnn::rnn_direction;
using platform::to_void_cast;

template <typename T>
class GRUFusedMKLDNNKernel : public framework::OpKernel<T> {
 struct SeqInfo {
   SeqInfo(int start, int length, int seq_idx)
        : start(start), length(length), seq_idx(seq_idx) {}
    int start;
    int length;
    int seq_idx;
 };

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
         	   "It must use CPUPlace.");                                     
                                                                                   
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();  
    auto null_memory_ = null_memory(mkldnn_engine);

    // Get Input/Output/Attr from op
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* h0 = ctx.Input<Tensor>("H0");
    auto* weight_h = ctx.Input<Tensor>("WeightH");
    auto* weight_x = ctx.Input<Tensor>("WeightX");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* hidden = ctx.Output<LoDTensor>("Hidden");
    bool is_reverse = ctx.Attr<bool>("is_reverse");
    bool is_bidir = ctx.Attr<bool>("is_bidirection");

    // Get dimensions for input / weight / bias / output
    std::vector<int> x_dims = vectorize2int(input->dims());    
    std::vector<int> weight_x_dims = vectorize2int(weight_x->dims());
    std::vector<int> weight_h_dims = vectorize2int(weight_h->dims());

    int D = is_bidir ? 2 : 1;              // 2 if bidirection
    int L = ctx.Attr<int>("stack_level");  // number of layers
    int G = 3;                             // number of gates, 3 for gru
    int I = x_dims[1];                     // input feature size
    int C = weight_h_dims[0];              // hidden state size 

    // Allocate memory for output
    T* hidden_data = static_cast<T*>(hidden->mutable_data<T>(ctx.GetPlace()));

    // Input x is N sequences (sentences) each with variable time steps (words)
    // Example: x  = (s1,  s2,  s3) i.e. N = 3
    //          s1 = (w1,  w2,  w3, w4)
    //          s2 = (w5,  w6,  w7, w8, w9)
    //          s3 = (w10, w11, w12)
    //  So x is actually (w1, w2, ..., w12)
    //  x->lod is (0, 4, 9, 12) where
    //    4 means s2 start from position 4 (i.e. w5) with len 5 (i.e. 9-4)
    //    9 means s3 start from position 9 (i.e. w10) with len 3 (i.e. 12 - 9)
    //    ...
    //  SeqLen is 12, i.e. total time steps (words) of x
    //  Batch size "N" is 3, i.e. 3 sequences in this batch
    int SeqLen = x_dims[0];
    LoD x_lods = input->lod();
    PADDLE_ENFORCE_EQ(x_lods.size(), 1UL, "Only support one level sequence now.");  
    const auto& x_lod = x_lods[0];
    int N = x_lod.size() - 1;
    
    // Save position/length/index of each sequence into seq_info[]
    // index (i.e. seq_id) is needed due to seq_info[] will be reordered later
    // Example: seq_info should be
    //          ((0,  4,  0),
    //           (4,  5,  1),
    //           (9,  3,  2))
    std::vector<SeqInfo> seq_info;
    for (size_t seq_id = 0; seq_id < x_lod.size() - 1; seq_id++) {
      int seq_len = x_lod[seq_id + 1] - x_lod[seq_id];
      seq_info.emplace_back(x_lod[seq_id], seq_len, seq_id);
    }
    
    // Sort all sequences in this batch via length. Longest first
    // Example: after sort, seq_info should be
    //          ((4, 5, 1),
    //           (0, 4, 0),
    //           (9, 3, 2))
    //   it means the longest sequence in x start from position 4 with 5 time
    //   steps (words) and index 1 (i.e. the 2nd sequence)
    std::sort(seq_info.begin(), seq_info.end(),
              [](SeqInfo a, SeqInfo b) { return a.length > b.length; });

    // For RNN, each time step (word) in one sequence (sentences) has to be
    // processed step by step. However, we can consolidate one time step in
    // several sequences into one "TimeStep Batch" (i.e. "TBatch") and feed
    // into RNN computing. Since those sequences may have different length
    // (number of time steps), we will use the maximum length as TBatch
    // Example:
    //           x = (W1,   W2,  W3,  W4,       <- s1
    //                W5,   W6,  W7,  W8,  W9,  <- s2
    //                W10, W11, W12)            <- s3
    //    after reorder (indicated by seq_info[]), it can be regarded as
    //           x = (W5,   W6,  W7,  W8,  W9,  <- s2
    //                W1,   W2,  W3,  W4,       <- s3
    //                W10, W11, W12)
    //    To feed into RNN, we need transpose it into TimeBatch with padding
    //    tbatch_x = (W5,   W1,  W10,           <- 1st tbatch
    //                W6,   W2,  W11,           <- 2nd tbatch
    //                W7,   W3,  W12,           <- 3rd tbatch
    //                W8,   W4,    0,           <- 4th tbatch with 0 padding
    //                W9,    0,    0)           <- 5th tbatch with 0 padding
    //    TBatch = 5 since the longest sequence (s2) has 5 time steps
    //    tbatch_x is with shape [TBatch, N, I] where x is [SeqLen, I]
    int TBatch = seq_info[0].length;
    PADDLE_ENFORCE_GT(TBatch, 0, "No data in input.");

    // Important! Don't mix "batch" of x (N = 3) with "tbatch" of RNN (TBatch=5)
  
    // We use a 3 level LoD to save relationship between x and tbatch_x
    //     tbatch_lods[0] = {0, 3,  6, 9, 11, 12}
    //     tbatch_lods[1] = {4, 0,  9,
    //                       5, 1, 10,
    //                       6, 2, 11,
    //                       7, 3,
    //                       8}
    //     tbatch_lods[2] = {1, 0, 2} 
    //
    // tbatch_lods[0] is the start position of each tbatch in tbatch_x
    // tbatch_lods[0][2] = 6 means the 3rd tbatch (w7,w3,w12) is at pos 6
    //
    // tbatch_lods[1] is the raw index in original input x ("0" based!)
    // tbatch_lods[1][7] = 2 means the 8th position of tbatch_x (W3) correspond to
    //                       the 3rd position of original x (W3)
    //
    // tbatch_lods[2] is the order of original setence, i.e. {s2, s1, s3}
    paddle::framework::LoD tbatch_lods;
    tbatch_lods.emplace_back(std::vector<size_t>{0});
    tbatch_lods.emplace_back(std::vector<size_t>{0});
    tbatch_lods.emplace_back(std::vector<size_t>{0});
    tbatch_lods[0].resize(static_cast<size_t>(TBatch+1));
    tbatch_lods[1].resize(static_cast<size_t>(SeqLen+1));
    tbatch_lods[2].resize(seq_info.size());

    // We use tbatch_lens to record the len of each tbatch
    // 0 < tbatch_lens[0..TBatch-1] <= N
    std::vector<int> tbatch_lens;
    tbatch_lens.resize(TBatch);
    
    // compute tbatch_lods and tbatch_lens which will be used in output
    size_t* tbatch_starts = tbatch_lods[0].data();
    size_t* tbatch2seq_idx = tbatch_lods[1].data();
    size_t* seq_order = tbatch_lods[2].data();
    tbatch_starts[0] = 0;
    for (int tb = 0; tb < TBatch; tb++) { // each TimeStep Batch
      auto offset = static_cast<int>(tbatch_starts[tb]);
      int seq;
      for (seq = 0; seq < N; ++seq) { // each sequence in current TimeStep batch
        int seq_len = seq_info[seq].length;
        int seq_start = seq_info[seq].start;
        if (tb < seq_len) {
          tbatch2seq_idx[offset] = seq_start + tb;
          offset++;
        } else {
          break;
        }
      }
      tbatch_starts[tb + 1] = static_cast<size_t>(offset);
      tbatch_lens[tb] = seq + 1;
    }
    for (size_t i = 0; i < seq_info.size(); ++i) {
      seq_order[i] = seq_info[i].seq_idx;
    }

    // reorder input x to tbatch_x 
    const T* x_data = input->data<T>();
    std::vector<T> tbatch_x(TBatch * N * I, 0); // unpack with zero padding
    T* tbatch_x_data = tbatch_x.data();
    for (int seq = 0; seq < N; seq++) { // for each sequence
      // get source address of this sequence
      auto ss = x_data + seq_info[seq].start * I;
      for (int tb = 0; tb < seq_info[seq].length; tb++) { // for each time step
        // get src/target address for this time step in this sequence
        auto sst = ss + tb * I;
        auto dst = tbatch_x_data + (tb * N + seq) * I;
	memcpy(dst, sst, I * sizeof(T));
      }
    }

    // Create mkldnn input memory with reordered data
    auto input_format = memory::format::tnc;
    auto input_md = MKLDNNMemDesc({TBatch, N, I},
		    MKLDNNGetDataType<T>(), input_format);
    auto input_memory_pd = memory::primitive_desc(input_md,
		    mkldnn_engine);
    auto input_mpd = memory::primitive_desc(input_md, mkldnn_engine);
    auto input_memory = memory(input_mpd, to_void_cast(tbatch_x_data));

    // Input h0
    auto h0_md = mkldnn::zero_md();
    const T* h0_data;
    auto h0_memory = null_memory_;
    if (h0) {
	std::vector<int> h0_dims = vectorize2int(h0->dims());
	// fixme: H0 in Paddle GRU op is (N x LDSC). Is reorder needed here?
	auto h0_format = memory::format::ldsnc;
	auto h0_md = MKLDNNMemDesc({L,D,1,N,C}, MKLDNNGetDataType<T>(), h0_format);
	auto h0_memory_pd = memory::primitive_desc(h0_md, mkldnn_engine);
	auto h0_mpd = memory::primitive_desc(h0_md, mkldnn_engine);
	h0_data = h0->data<T>();
        h0_memory = memory(h0_mpd, to_void_cast(h0_data));
    }

    // Weight W_x
    // fixme: WeightX in Paddle is (C x L*D*G*C). Is reorder needed here?
    auto weight_x_md = MKLDNNMemDesc({L, D, I, G, C}, MKLDNNGetDataType<T>(),
		    memory::format::ldigo);
    auto weight_x_memory_pd = memory::primitive_desc(weight_x_md,
		    mkldnn_engine);
    const T* weight_x_data = weight_x->data<T>();
    auto weight_x_memory = memory(weight_x_memory_pd,
		    to_void_cast(weight_x_data));

    // Weight W_h
    // ToBeDone! WeightH in Paddle is split into 2 parts, i.e. weights of
    // the update gate and reset gate with shape (D x 2D), and the second
    // part is weights of output candidate with shape (D x D)
    // The two parts of memory need be consolidated into one continuous
    // one before passing to MKLDNN
    auto weight_h_md = MKLDNNMemDesc({L, D, C, G, C}, MKLDNNGetDataType<T>(),
		    memory::format::ldigo);
    auto weight_h_memory_pd = memory::primitive_desc(weight_h_md,
		    mkldnn_engine);
    const T* weight_h_data = weight_h->data<T>();
    auto weight_h_memory = memory(weight_h_memory_pd,
		    to_void_cast(weight_h_data));

    // Bias
    // fixme: Bias in Paddle is (1 x L*D*G*C). Is reorder needed here?        
    auto bias_md = MKLDNNMemDesc({L, D, G, C}, MKLDNNGetDataType<T>(),
		    memory::format::ldgo);
    auto bias_memory_pd = memory::primitive_desc(bias_md, mkldnn_engine);
    const T* bias_data = bias->data<T>();
    auto bias_memory = memory(bias_memory_pd, to_void_cast(bias_data));

    // Hidden h (output)
    auto hidden_md = MKLDNNMemDesc({TBatch, N, C}, MKLDNNGetDataType<T>(),
                 		   memory::format::tnc);

    // create GRU forward primitive desc
    auto cell = rnn_cell::desc(mkldnn::algorithm::vanilla_gru);
    rnn_direction direction;
    if (is_bidir) {
      // Fix me: now hardcode "concat". Should add "sum" also
      direction = rnn_direction::bidirectional_concat;
    } else {
      direction = is_reverse ?
	rnn_direction::unidirectional_right2left :
	rnn_direction::unidirectional_left2right;
    }
    auto forward_desc = rnn_forward::desc(
		    mkldnn::prop_kind::forward_inference,
		    cell,
		    direction,
		    input_md,
		    h0_md,
		    weight_x_md,
		    weight_h_md,
		    bias_md,
		    hidden_md,
		    mkldnn::zero_md());
    auto forward_pd = rnn_forward::primitive_desc(forward_desc, mkldnn_engine);

    // create dest memory (TBatch,N,C) for GRU forward
    auto tbatch_hidden_memory = mkldnn::memory(forward_pd.dst_layer_primitive_desc());
    auto forward_op = rnn_forward(
		    forward_pd,
		    input_memory,
		    h0_memory,
		    weight_x_memory,
		    weight_h_memory,
		    bias_memory,
		    tbatch_hidden_memory,
		    null_memory_,		    
		    null_memory_);

    std::vector<mkldnn::primitive> pipeline = {forward_op};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    // reorder batch hidden data (TBatch,N,C) back to hidden data (SeqLen,C)
    T *tbatch_hidden_data = static_cast<T*>(tbatch_hidden_memory.get_data_handle());
    int offset = 0;
    for (int tb = 0; tb < TBatch; tb++) { // for each time step
      for (int seq = 0; seq < tbatch_lens[tb]; seq++) {
	// for each word at specified time step
	auto dst = hidden_data + offset * C;
	auto src = tbatch_hidden_data + (tb * N + seq) * C;
	memcpy(dst, src, C * sizeof(T));
      }
      offset += tbatch_lens[tb];
    }
    PADDLE_ENFORCE_EQ(offset, SeqLen,
		      "Hidden output should have same length as input x");

    // Need set LoD to output tensor
    hidden->set_lod(tbatch_lods);
    
    hidden->set_layout(DataLayout::kMKLDNN);
    hidden->set_format((const mkldnn::memory::format)tbatch_hidden_memory.get_primitive_desc().desc().data.format);
    //hidden->set_format(GetMKLDNNFormat(hidden_state_memory));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(gru_fused, MKLDNN, ::paddle::platform::CPUPlace,
	ops::GRUFusedMKLDNNKernel<float>)                           
