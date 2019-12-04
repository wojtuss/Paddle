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
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

DEFINE_int32(warmup_iterations, 1, "Number of warmup iterations.");
DEFINE_bool(verbose, false, "Print output for each iteration.");

namespace paddle {
namespace inference {
namespace analysis {

using framework::DataTypeTrait;
using framework::Tensor;
using framework::make_ddim;
using framework::vectorize;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::MKLDNNGetDataType;
using platform::MKLDNNMemDesc;
using platform::CPUPlace;
using platform::SetNumThreads;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::stream;
using mkldnn::prop_kind;

double run(inner_product_forward* ip) {
  SetNumThreads(FLAGS_num_threads);
  Timer run_timer;
  double elapsed_time = 0;
  double iter_time = 0;
  std::vector<mkldnn::primitive> pipeline{*ip};
  for (int i = 0; i < FLAGS_warmup_iterations; ++i) {
    run_timer.tic();
    stream(stream::kind::eager).submit(pipeline).wait();
    iter_time = run_timer.toc();
    if (FLAGS_verbose) {
      LOG(INFO) << "--- Warmup iteration " << i << ", latency: " << iter_time
                << " ---";
    }
  }
  for (int i = 0; i < FLAGS_iterations; ++i) {
    run_timer.tic();
    stream(stream::kind::eager).submit(pipeline).wait();
    iter_time = run_timer.toc();
    elapsed_time += iter_time;
    if (FLAGS_verbose) {
      LOG(INFO) << "--- Iteration " << i << ", latency: " << iter_time << " ---"
                << std::endl;
    }
  }
  // return batch latency
  return elapsed_time / FLAGS_iterations;
}

template <typename Tin, typename Tw, typename Tout>
double create_and_run_inner_product(std::initializer_list<int> dims_in,
                                    memory::format f_in,
                                    std::initializer_list<int> dims_w,
                                    memory::format f_w,
                                    std::initializer_list<int> dims_out) {
  auto dev_ctx = MKLDNNDeviceContext(CPUPlace());
  auto mkldnn_engine = dev_ctx.GetEngine();

  auto input = Tensor(DataTypeTrait<Tin>::DataType());
  input.set_format(f_in);
  auto input_dims = make_ddim(dims_in);
  input.mutable_data<Tin>(input_dims, CPUPlace());

  auto w = Tensor(DataTypeTrait<Tw>::DataType());
  w.set_format(f_w);
  auto w_dims = make_ddim(dims_w);
  w.mutable_data<Tw>(w_dims, CPUPlace());

  auto output = Tensor(DataTypeTrait<Tout>::DataType());
  auto output_dims = make_ddim(dims_out);
  output.Resize(output_dims);

  std::vector<int> fc_src_tz = vectorize<int>(input.dims());
  std::vector<int> fc_weights_tz = vectorize<int>(w.dims());
  std::vector<int> fc_dst_tz = vectorize<int>(output.dims());

  auto fc_src_md =
      MKLDNNMemDesc(fc_src_tz, MKLDNNGetDataType<Tin>(), input.format());
  auto fc_src_memory_pd = memory::primitive_desc(fc_src_md, mkldnn_engine);
  auto fc_src_memory =
      memory(fc_src_memory_pd, to_void_cast<Tin>(input.data<Tin>()));

  auto fc_weights_md =
      MKLDNNMemDesc(fc_weights_tz, MKLDNNGetDataType<Tw>(), w.format());
  auto fc_weights_memory_pd =
      memory::primitive_desc(fc_weights_md, mkldnn_engine);
  auto fc_weights_memory =
      memory(fc_weights_memory_pd, to_void_cast<Tw>(w.data<Tw>()));

  auto fc_dst_md = MKLDNNMemDesc(fc_dst_tz, mkldnn::memory::f32,
                                 mkldnn::memory::format::any);

  std::shared_ptr<inner_product_forward::desc> fc_desc_p;
  fc_desc_p.reset(new inner_product_forward::desc(prop_kind::forward, fc_src_md,
                                                  fc_weights_md, fc_dst_md));
  auto fc_prim_desc =
      inner_product_forward::primitive_desc(*fc_desc_p, mkldnn_engine);

  auto fc_dst_memory_pd = fc_prim_desc.dst_primitive_desc();
  auto fc_dst_memory_sz = fc_dst_memory_pd.get_size();
  Tout* output_data = output.mutable_data<Tout>(CPUPlace(), fc_dst_memory_sz);

  auto fc_dst_memory =
      memory(fc_dst_memory_pd, to_void_cast<Tout>(output_data));

  auto ip = inner_product_forward(fc_prim_desc, fc_src_memory,
                                  fc_weights_memory, fc_dst_memory);

  return run(&ip);
}

/*
 * TEST(Mkldnn_gemm_uint8, benchmark) {
 *   LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128
 * "
 *                "and input data type uint8_t ===";
 *
 *   auto input_dims = {1, 128, 768, 768};
 *   auto input_format = memory::nchw;
 *   auto weights_dims = {128, 128, 768, 768};
 *   auto weights_format = memory::oihw;
 *   auto output_dims = {1, 128};
 *
 *   double latency1 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency1, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency without padding: " << latency1;
 *
 *   double latency2 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency2, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency with padding: " << latency2;
 *   LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
 * }
 */

TEST(Mkldnn_gemm_3D_uint8_oiw, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type uint8_t ===";

  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128, 128, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128 + 4, 128 + 4, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

TEST(Mkldnn_gemm_3D_int8_oiw, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type int8_t ===";
  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128, 128, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128 + 4, 128 + 4, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

TEST(Mkldnn_gemm_3D_float_oiw, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type float ===";
  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128, 128, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<float, float, float>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768};
    auto input_format = memory::ncw;
    auto weights_dims = {128 + 4, 128 + 4, 768};
    auto weights_format = memory::oiw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<float, float, float>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

TEST(Mkldnn_gemm_4D_uint8, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type uint8_t ===";
  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128, 128, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128 + 4, 128 + 4, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<uint8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

TEST(Mkldnn_gemm_4D_int8, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type int8_t ===";
  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128, 128, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128 + 4, 128 + 4, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

TEST(Mkldnn_gemm_4D_float, benchmark) {
  LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128 "
               "and input data type float ===";
  double latency1, latency2;

  {
    auto input_dims = {1, 128, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128, 128, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128};

    latency1 = create_and_run_inner_product<float, float, float>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency1, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency without padding: " << latency1;
  }

  {
    auto input_dims = {1, 128 + 4, 768, 768};
    auto input_format = memory::nchw;
    auto weights_dims = {128 + 4, 128 + 4, 768, 768};
    auto weights_format = memory::oihw;
    auto output_dims = {1, 128 + 4};

    latency2 = create_and_run_inner_product<float, float, float>(
        input_dims, input_format, weights_dims, weights_format, output_dims);
    EXPECT_GT(latency2, 0);
    LOG(INFO) << "Iterations: " << FLAGS_iterations
              << ", average latency with padding: " << latency2;
  }

  LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
}

/*
 * TEST(Mkldnn_gemm_int8, benchmark) {
 *   LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128
 * "
 *                "and input data type int8_t ===";
 *
 *   auto input_dims = {1, 128, 768, 768};
 *   auto input_format = memory::nchw;
 *   auto weights_dims = {128, 128, 768, 768};
 *   auto weights_format = memory::oihw;
 *   auto output_dims = {1, 128};
 *
 *   double latency1 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency1, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency without padding: " << latency1;
 *
 *   double latency2 = create_and_run_inner_product<int8_t, int8_t, int32_t>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency2, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency with padding: " << latency2;
 *   LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
 * }
 */

/*
 * TEST(Mkldnn_gemm_int8_fp32, benchmark) {
 *   LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128
 * "
 *                "and input data type int8_t ===";
 *
 *   auto input_dims = {1, 128, 768, 768};
 *   auto input_format = memory::nchw;
 *   auto weights_dims = {128, 128, 768, 768};
 *   auto weights_format = memory::oihw;
 *   auto output_dims = {1, 128};
 *
 *   double latency1 = create_and_run_inner_product<int8_t, int8_t, float>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency1, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency without padding: " << latency1;
 *
 *   double latency2 = create_and_run_inner_product<int8_t, int8_t, float>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency2, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency with padding: " << latency2;
 *   LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
 * }
 *
 * TEST(Mkldnn_gemm_fp32, benchmark) {
 *   LOG(INFO) << "=== Benchmarking MKL-DNN inner product with channels size 128
 * "
 *                "and input data type float ===";
 *
 *   auto input_dims = {1, 128, 768, 768};
 *   auto input_format = memory::nchw;
 *   auto weights_dims = {128, 128, 768, 768};
 *   auto weights_format = memory::oihw;
 *   auto output_dims = {1, 128};
 *
 *   double latency1 = create_and_run_inner_product<float, float, float>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency1, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency without padding: " << latency1;
 *
 *   double latency2 = create_and_run_inner_product<float, float, float>(
 *       input_dims, input_format, weights_dims, weights_format, output_dims);
 *   EXPECT_GT(latency2, 0);
 *   LOG(INFO) << "Iterations: " << FLAGS_iterations
 *             << ", average latency with padding: " << latency2;
 *   LOG(INFO) << "Latency with/without padding ratio: " << latency1 / latency2;
 * }
 */

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
