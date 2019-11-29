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
#include "paddle/fluid/platform/mkldnn_helper.h"

DEFINE_int32(warmup_iterations, 1, "Number of warmup iterations.");

namespace paddle {
namespace inference {
namespace analysis {

using framework::DataLayout;
using framework::Tensor;
using framework::make_ddim;
using framework::vectorize;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using platform::MKLDNNMemDesc;
using platform::CPUPlace;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

TEST(Gemm_int8_padding, benchmark) {
  auto dev_ctx = MKLDNNDeviceContext(CPUPlace());
  auto mkldnn_engine = dev_ctx.GetEngine();

  auto input = Tensor(framework::proto::VarType::UINT8);
  // input.set_format(memory::ncw);
  // auto input_dims = make_ddim({1, 128, 768});
  input.set_format(memory::nchw);
  auto input_dims = make_ddim({1, 128, 768, 768});
  input.mutable_data<uint8_t>(input_dims, CPUPlace());

  auto w = Tensor(framework::proto::VarType::INT8);
  // w.set_format(memory::oiw);
  // auto w_dims = make_ddim({128, 128, 768});
  w.set_format(memory::oihw);
  auto w_dims = make_ddim({128, 128, 768, 768});
  w.mutable_data<int8_t>(w_dims, CPUPlace());

  auto output = Tensor(framework::proto::VarType::UINT8);
  auto output_dims = make_ddim({1, 128});
  output.Resize(output_dims);
  // output.mutable_data<uint8_t>(output_dims, CPUPlace());

  std::vector<int> fc_src_tz = vectorize<int>(input.dims());
  std::vector<int> fc_weights_tz = vectorize<int>(w.dims());
  std::vector<int> fc_dst_tz = vectorize<int>(output.dims());

  auto fc_src_md =
      MKLDNNMemDesc(fc_src_tz, MKLDNNGetDataType<uint8_t>(), input.format());
  auto fc_src_memory_pd = memory::primitive_desc(fc_src_md, mkldnn_engine);
  auto fc_src_memory =
      memory(fc_src_memory_pd, to_void_cast<uint8_t>(input.data<uint8_t>()));

  auto fc_weights_md =
      MKLDNNMemDesc(fc_weights_tz, MKLDNNGetDataType<int8_t>(), w.format());
  auto fc_weights_memory_pd =
      memory::primitive_desc(fc_weights_md, mkldnn_engine);
  auto fc_weights_memory =
      memory(fc_weights_memory_pd, to_void_cast<int8_t>(w.data<int8_t>()));

  auto fc_dst_md = MKLDNNMemDesc(fc_dst_tz, mkldnn::memory::f32,
                                 mkldnn::memory::format::any);

  std::shared_ptr<inner_product_forward::desc> fc_desc_p;
  fc_desc_p.reset(new inner_product_forward::desc(prop_kind::forward, fc_src_md,
                                                  fc_weights_md, fc_dst_md));
  auto fc_prim_desc =
      inner_product_forward::primitive_desc(*fc_desc_p, mkldnn_engine);

  auto fc_dst_memory_pd = fc_prim_desc.dst_primitive_desc();
  auto fc_dst_memory_sz = fc_dst_memory_pd.get_size();
  int32_t* output_data =
      output.mutable_data<int32_t>(CPUPlace(), fc_dst_memory_sz);

  auto fc_dst_memory =
      memory(fc_dst_memory_pd, to_void_cast<int32_t>(output_data));

  auto fc = inner_product_forward(fc_prim_desc, fc_src_memory,
                                  fc_weights_memory, fc_dst_memory);

  Timer run_timer;
  double elapsed_time = 0;
  double iter_time = 0;
  std::vector<mkldnn::primitive> pipeline{fc};
  for (int i = 0; i < FLAGS_warmup_iterations; ++i) {
    run_timer.tic();
    stream(stream::kind::eager).submit(pipeline).wait();
    iter_time = run_timer.toc();
    LOG(INFO) << "--- Warmup iteration " << i << ", latency: " << iter_time
              << " ---";
  }
  for (int i = 0; i < FLAGS_iterations; ++i) {
    // push primitive to stream and wait until it's executed
    run_timer.tic();
    stream(stream::kind::eager).submit(pipeline).wait();
    iter_time = run_timer.toc();
    elapsed_time += iter_time;
    LOG(INFO) << "--- Iteration " << i << ", latency: " << iter_time << " ---"
              << std::endl;
  }
  auto batch_latency = elapsed_time / FLAGS_iterations;
  LOG(INFO) << "====== Iterations: " << FLAGS_iterations
            << ", average latency: " << batch_latency << " ======";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
