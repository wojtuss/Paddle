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

#include "paddle/fluid/operators/gru_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::MKLDNNDeviceContext;


using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::CPUDeviceContext;
using paddle::platform::MKLDNNMemDesc;
using paddle::platform::MKLDNNGetDataType;
using framework::DataLayout;
using framework::vectorize2int;
using mkldnn::memory;
using mkldnn::rnn_cell;
using mkldnn::rnn_forward;
using platform::to_void_cast;


template <typename T>
class GRUMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::cout << "--- GRUMKLDNNKernel::Compute ---\n";
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
		    "It must use CPUPlace.");                                     
                                                                                   
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    auto null_memory_ = null_memory(mkldnn_engine);
    // dimension constants
    const int l = 1; // number of GRU layers
    const int d = 1; // 2 for bidirectional, 1 otherwise
    const int g = 3; // number of gates, 3 for gru

    auto* input = ctx.Input<LoDTensor>("Input");
    auto* h0 = ctx.Input<Tensor>("H0");
    auto* weight_h = ctx.Input<Tensor>("Weight");
    auto* weight_x = ctx.Input<Tensor>("WeightX");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* hidden = ctx.Output<LoDTensor>("Hidden");
    hidden->mutable_data<T>(ctx.GetPlace());
    bool is_reverse = ctx.Attr<bool>("is_reverse");

    // Input x
    std::vector<int> in_dims = vectorize2int(input->dims());
    // auto input_format = input->format();
    auto input_format = memory::format::tnc;
    int t = in_dims[0];
    int n = 1;	// TODO: check if Paddle can provide batch mode processing
    int c = in_dims[1];
    auto input_md = MKLDNNMemDesc({t, n, c},
		    MKLDNNGetDataType<T>(), input_format);
    auto input_memory_pd = memory::primitive_desc(input_md,
		    mkldnn_engine);
    auto input_mpd = memory::primitive_desc(input_md, mkldnn_engine);
    const T* input_data = input->data<T>();
    auto input_memory = memory(input_mpd, to_void_cast(input_data));

    // Input h0
    auto h0_md = mkldnn::zero_md();
    const T* h0_data;
    // auto h0_memory = mkldnn::null_memory(cpu_engine);
    auto h0_memory = null_memory_;
    if (h0) {
	std::vector<int> h0_dims = vectorize2int(h0->dims());
	auto h0_format = memory::format::ldsnc;
	auto h0_md = MKLDNNMemDesc(in_dims, MKLDNNGetDataType<T>(), h0_format);
	auto h0_memory_pd = memory::primitive_desc(h0_md, mkldnn_engine);
	auto h0_mpd = memory::primitive_desc(h0_md, mkldnn_engine);
	h0_data = h0->data<T>();
        h0_memory = memory(h0_mpd, to_void_cast(h0_data));
    }

    // Weight W_x
    std::vector<int> weight_x_dims = vectorize2int(weight_x->dims());
    int i = weight_x_dims[1];
    int o = weight_x_dims[0];
    auto weight_x_md = MKLDNNMemDesc(weight_x_dims, MKLDNNGetDataType<T>(),
		    memory::format::ldigo);
    auto weight_x_memory_pd = memory::primitive_desc(weight_x_md,
		    mkldnn_engine);
    const T* weight_x_data = weight_x->data<T>();
    auto weight_x_memory = memory(weight_x_memory_pd,
		    to_void_cast(weight_x_data));

    // Weight W_h
    std::vector<int> weight_h_dims = vectorize2int(weight_h->dims());
    i = weight_h_dims[1];
    o = weight_h_dims[0];
    
    auto weight_h_md = MKLDNNMemDesc({l, d, i, g, o}, MKLDNNGetDataType<T>(),
		    memory::format::ldigo);
    auto weight_h_memory_pd = memory::primitive_desc(weight_h_md,
		    mkldnn_engine);
    const T* weight_h_data = weight_h->data<T>();
    auto weight_h_memory = memory(weight_h_memory_pd,
		    to_void_cast(weight_h_data));

    // Bias
    std::vector<int> bias_dims = vectorize2int(bias->dims());
    o = bias_dims[0];
    auto bias_md = MKLDNNMemDesc({l, d, g, o}, MKLDNNGetDataType<T>(),
		    memory::format::ldgo);
    auto bias_memory_pd = memory::primitive_desc(bias_md, mkldnn_engine);
    const T* bias_data = bias->data<T>();
    auto bias_memory = memory(bias_memory_pd, to_void_cast(bias_data));

    // Hidden h (output)
    // auto out_dims_v = h0->dims();
    std::vector<int> hidden_dims = vectorize2int(hidden->dims());
    auto hidden_format = memory::format::tnc;
    t = hidden_dims[0];
    n = 1;	// TODO: check if Paddle can provide batch mode processing
    c = hidden_dims[1];
    auto hidden_md = MKLDNNMemDesc({t, n, c}, MKLDNNGetDataType<T>(),
		    hidden_format);

    auto cell = rnn_cell::desc(mkldnn::algorithm::vanilla_gru);
    auto direction = is_reverse ?
	    mkldnn::rnn_direction::unidirectional_right2left :
	    mkldnn::rnn_direction::unidirectional_left2right;
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
    auto forward_memory = mkldnn::memory(forward_pd.workspace_primitive_desc());
    auto forward_op = rnn_forward(
		    forward_pd,
		    input_memory,
		    h0_memory,
		    weight_x_memory,
		    weight_h_memory,
		    bias_memory,
		    forward_memory,
		    null_memory_,
		    null_memory_);

    std::vector<mkldnn::primitive> pipeline = {forward_op};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    hidden->set_layout(DataLayout::kMKLDNN);
    hidden->set_format(hidden_format);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(gru, MKLDNN, ::paddle::platform::CPUPlace,
	ops::GRUMKLDNNKernel<float>)                           
                                                                                 

