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
    //int TSum = x_dims[0];                  // total time steps in mini-batch
    int I = x_dims[1];                     // input feature size
    int C = weight_h_dims[0];              // hidden state size 

    // Allocate memory for output
    hidden->mutable_data<T>(ctx.GetPlace());

    // Input x is a batch of sequence each with variable length (time step)
    // Go through all sequences and get start position and len for each seq
    LoD x_lods = input->lod();
    PADDLE_ENFORCE_EQ(x_lods.size(), 1UL, "Only support one level sequence now.");  
    const auto& x_lod = x_lods[0];
    
    std::vector<SeqInfo> seq_info;
    for (size_t seq_id = 0; seq_id < x_lod.size() - 1; seq_id++) {
      int seq_len = x_lod[seq_id + 1] - x_lod[seq_id];
      seq_info.emplace_back(x_lod[seq_id], seq_len, seq_id);
    }

    // Now we know batch size N
    int N = seq_info.size();
    
    // Sort all sequences in this batch via length
    std::sort(seq_info.begin(), seq_info.end(),
              [](SeqInfo a, SeqInfo b) { return a.length > b.length; });

    // Now we know the maximum lengh of all sequences in this batch
    int TMax = seq_info[0].length;
    PADDLE_ENFORCE_GT(TMax, 0, "No data in input.");

    // Need reorder input x from [TSum, I] to [TMax, N, I]
    //   TSum: total time steps in this batch
    //   TMax: max lengh of all sequences in this batch
    //      N: batch size
    //      I: input feature size
    // Two operations need here
    //  1. Unpack: each sequence will be extended to TMax(*I)
    //  2. Transpose: T1 of all sequences firstly, then T2, T3, ...
    const T* x_data = input->data<T>();
    std::vector<T> batch_x(TMax * N * I, 0); // unpack with zero padding
    T* batch_x_data = batch_x.data();
#pragma omp parallel for simd
    for (int s = 0; s < N; s++) { // for each sequence
      // get source address of this sequence
      auto ss = x_data + seq_info[s].start * I;
      for (int t = 0; t < seq_info[s].length; t++) { // for each time step
	// get src/target address for this time step in this sequence
	auto sst = ss + t * I;
	auto dst = batch_x_data + t * N * I + s;
        for (int c = 0; c < I; c++) {
          *(dst + c) = *(sst + c); // memcpy instead?
	}
      }
    }

    // Create mkldnn input memory with reordered data
    auto input_format = memory::format::tnc;
    auto input_md = MKLDNNMemDesc({TMax, N, I},
		    MKLDNNGetDataType<T>(), input_format);
    auto input_memory_pd = memory::primitive_desc(input_md,
		    mkldnn_engine);
    auto input_mpd = memory::primitive_desc(input_md, mkldnn_engine);
    auto input_memory = memory(input_mpd, to_void_cast(batch_x_data));

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
    auto hidden_md = MKLDNNMemDesc({TMax, N, C}, MKLDNNGetDataType<T>(),
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

    // create dest memory (TMax,N,C) for GRU forward
    auto batch_hidden_memory = mkldnn::memory(forward_pd.dst_layer_primitive_desc());
    auto forward_op = rnn_forward(
		    forward_pd,
		    input_memory,
		    h0_memory,
		    weight_x_memory,
		    weight_h_memory,
		    bias_memory,
		    batch_hidden_memory,
		    null_memory_,		    
		    null_memory_);

    std::vector<mkldnn::primitive> pipeline = {forward_op};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    // reorder batch hidden memory (TMax,N,C) back to a LoD output (TSum,C)

    hidden->set_layout(DataLayout::kMKLDNN);
    hidden->set_format((const mkldnn::memory::format)batch_hidden_memory.get_primitive_desc().desc().data.format);
    //hidden->set_format(GetMKLDNNFormat(hidden_state_memory));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(gru_fused, MKLDNN, ::paddle::platform::CPUPlace,
	ops::GRUFusedMKLDNNKernel<float>)                           
                                                                                 

