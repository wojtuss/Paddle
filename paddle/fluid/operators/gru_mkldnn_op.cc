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

template <typename T>
class GRUMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
	std::cout << "--- GRUMKLDNNKernel::Compute ---\n";
	PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
			"It must use CPUPlace.");                                     
                                                                                   
	// auto& dev_ctx =
		// ctx.template device_context<MKLDNNDeviceContext>();    
		// const auto& mkldnn_engine = dev_ctx.GetEngine();                             
  }
};

template <typename T>
class GRUMKLDNNGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::cout << "--- GRUMKLDNNGradKernel::Compute ---\n";
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(gru, MKLDNN, ::paddle::platform::CPUPlace,
	ops::GRUMKLDNNKernel<float>)                           
                                                                                 
// REGISTER_OP_KERNEL(gru_grad, MKLDNN, ::paddle::platform::CPUPlace,   
	// ops::GRUMKLDNNGradKernel<float>)

