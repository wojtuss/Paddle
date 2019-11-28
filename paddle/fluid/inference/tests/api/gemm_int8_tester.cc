/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::DataLayout;
using framework::Tensor;
using framework::make_ddim;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

TEST(Gemm_int8_padding, benchmark) {
  auto input = Tensor(proto::VarType::UINT8);
  auto input_dims = make_ddim({1, 128, 768});
  input.mutable_data<uint8_t>(input_dims, platform::CPUPlace);

  auto w = Tensor(proto::VarType::INT8);
  auto w_dims = make_ddim({768, 768});
  w.mutable_data<int8_t>(w_dims, platform::CPUPlace);

  auto output = Tensor(proto::VarType::UINT8);
  auto output_dims = make_ddim({1, 128, 768});
  output.Resize(output_dims);
  // output.mutable_data<uint8_t>(output_dims, platform::CPUPlace);

  std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());
  std::vector<int> weights_tz = paddle::framework::vectorize2int(w->dims());
  std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());

  auto fc_src_md = platform::MKLDNNMemDesc(
      fc_src_tz, platform::MKLDNNGetDataType<T>(), input->format());
  auto fc_src_memory_pd = memory::primitive_desc(fc_src_md, mkldnn_engine);
  auto fc_src_memory =
      memory(fc_src_memory_pd, to_void_cast<T>(input->data<T>()));

  auto fc_weights_md = platform::MKLDNNMemDesc(
      fc_weights_tz, platform::MKLDNNGetDataType<T>(), w->format());
  auto fc_weights_memory_pd =
      memory::primitive_desc(fc_weights_md, mkldnn_engine);
  auto fc_weights_memory =
      memory(fc_weights_memory_pd, to_void_cast<T>(w->data<T>()));

  auto fc_dst_md = platform::MKLDNNMemDesc(fc_dst_tz, mkldnn::memory::f32,
                                           mkldnn::memory::format::any);

  std::shared_ptr<memory> fc_bias_memory_p;
  std::shared_ptr<inner_product_forward::desc> fc_desc_p;
  if (bias) {
    std::vector<int> fc_bias_tz =
        paddle::framework::vectorize2int(bias->dims());
    auto fc_bias_md = platform::MKLDNNMemDesc(
        fc_bias_tz, platform::MKLDNNGetDataType<T>(), bias->format());
    auto fc_bias_memory_pd = memory::primitive_desc(fc_bias_md, mkldnn_engine);
    fc_bias_memory_p.reset(
        new memory(fc_bias_memory_pd, to_void_cast<T>(bias->data<T>())));

    fc_desc_p.reset(new inner_product_forward::desc(
        prop_kind::forward, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md));
  } else {
    fc_desc_p.reset(new inner_product_forward::desc(
        prop_kind::forward, fc_src_md, fc_weights_md, fc_dst_md));
  }
  auto fc_prim_desc =
      inner_product_forward::primitive_desc(*fc_desc_p, mkldnn_engine);

  auto fc_dst_memory_pd = fc_prim_desc.dst_primitive_desc();
  auto fc_dst_memory_sz = fc_dst_memory_pd.get_size();
  T* output_data = output->mutable_data<T>(ctx.GetPlace(), fc_dst_memory_sz);

  auto fc_dst_memory = memory(fc_dst_memory_pd, to_void_cast<T>(output_data));

  auto fc = bias ? inner_product_forward(fc_prim_desc, fc_src_memory,
                                         fc_weights_memory, *fc_bias_memory_p,
                                         fc_dst_memory)
                 : inner_product_forward(fc_prim_desc, fc_src_memory,
                                         fc_weights_memory, fc_dst_memory);

  // push primitive to stream and wait until it's executed
  //     std::vector<mkldnn::primitive> pipeline{fc};
  //         stream(stream::kind::eager).submit(pipeline).wait();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
