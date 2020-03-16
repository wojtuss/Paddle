/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

enum { kMATMUL_DNNL_FP32 = 1, kMATMUL_DNNL_INT8 = 2 };

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
// using mkldnn::matmul;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename Tx, typename Ty, typename Tout>
class MatMulPrimitiveFactory {
 public:
  explicit MatMulPrimitiveFactory(const mkldnn::engine& engine)
      : engine_(engine) {}

  void ExecuteMatMulPrimitive(const LoDTensor* x, const LoDTensor* y,
                              LoDTensor* output, const ExecutionContext& ctx) {
    // If primitive has already been created and cached, don't create new one,
    // but update input and output data pointers and return it.
    if (matmul_) {
      UpdateDataPointers(ctx, x, y, output);
      this->Execute();
      return;
    }  // Otherwise, create a new one.

    boost::optional<mkldnn::matmul::primitive_desc> matmul_pd;
    matmul_pd = CreateMatMulPrimDescriptor(x, y, output, ctx);

    x_ = CreateMemory<Tx>(matmul_pd->src_desc(), x);
    y_ = CreateMemory<Ty>(matmul_pd->src_desc(), y);
    output_ = CreateDstMemory(*matmul_pd, ctx, output);

    // Return MKL-DNN primitive ready to be fed into pipeline and executed
    matmul_ = mkldnn::matmul(*matmul_pd);
    this->Execute();
  }

  void Execute() {
    mkldnn::stream astream(engine_);
    matmul_->execute(astream, {{MKLDNN_ARG_SRC, *x_},
                               {MKLDNN_ARG_WEIGHTS, *y_},
                               {MKLDNN_ARG_DST, *output_}});
    astream.wait();
  }

 private:
  void UpdateDataPointers(const ExecutionContext& ctx, const Tensor* x,
                          const Tensor* y, Tensor* out) {
    x_->set_data_handle(to_void_cast(x->data<Tx>()));
    y_->set_data_handle(to_void_cast(y->data<Ty>()));
    output_->set_data_handle(out->mutable_data<Tout>(ctx.GetPlace()));
    // If the primitive exists, but the output tensor has changed its
    // variable, update its format to what has been determined in first
    // call to CreateFcPrimitive method.
    if (out->format() == MKLDNNMemoryFormat::undef) {
      out->set_format(x->format());
    }
  }

  mkldnn::matmul::primitive_desc CreateMatMulPrimDescriptor(
      const LoDTensor* x, const LoDTensor* y, LoDTensor* output,
      const ExecutionContext& ctx) {
    auto x_md = CreateMemDescriptor<Tx>(x, x->format());
    auto y_md = CreateMemDescriptor<Ty>(y, y->format());
    auto dst_md = CreateMemDescriptor<Tout>(output, MKLDNNMemoryFormat::any);
    const mkldnn::primitive_attr attrs;
    auto matmul_desc =
        mkldnn::matmul::desc(prop_kind::forward_scoring, x_md, y_md, dst_md);

    return mkldnn::matmul::primitive_desc(matmul_desc, attrs, engine_);
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(
      const std::vector<int64_t>& dims, MKLDNNMemoryFormat format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                  MKLDNNMemoryFormat format) {
    auto dims = framework::vectorize(tensor->dims());
    return CreateMemDescriptor<T>(dims, format);
  }

  template <typename T>
  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return memory(desc, engine_, platform::to_void_cast<T>(tensor->data<T>()));
  }

  // Create output memory based on output tensor and inner_product
  // primitive descriptor format chosen for output
  mkldnn::memory CreateDstMemory(
      const mkldnn::matmul::primitive_desc& matmul_pd,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_desc = matmul_pd.dst_desc();
    auto buffer_size = dst_desc.get_size();
    Tout* output_data = output->mutable_data<Tout>(ctx.GetPlace(), buffer_size);
    memory dst_mem(dst_desc, engine_, to_void_cast<Tout>(output_data));
    output->set_format(ctx.Input<LoDTensor>("X")->format());

    return dst_mem;
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> x_;
  boost::optional<memory> y_;
  boost::optional<memory> output_;
  boost::optional<mkldnn::matmul> matmul_;
};

// Attempt to fetch cached primitive factory based on provided parameters
// of input format, weight dimensions and output name.
// If not cached, create a new one.
template <typename Tx, typename Ty, typename Tout>
static std::shared_ptr<MatMulPrimitiveFactory<Tx, Ty, Tout>>
GetPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                    const ExecutionContext& ctx, const Tensor* x,
                    const Tensor* y, const mkldnn::engine& mkldnn_engine) {
  const std::string key =
      platform::CreateKey(platform::ThreadIDasStr(), x->format(),
                          framework::vectorize<int>(x->dims()), y->format(),
                          y->dims()[0], ctx.OutputName("Out"));

  auto prim_creator =
      std::static_pointer_cast<MatMulPrimitiveFactory<Tx, Ty, Tout>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<MatMulPrimitiveFactory<Tx, Ty, Tout>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename Tx, typename Ty>
static void ExecuteMatMul(const MKLDNNDeviceContext& dev_ctx,
                          const ExecutionContext& ctx, const LoDTensor* x,
                          const LoDTensor* y, LoDTensor* output,
                          const mkldnn::engine& mkldnn_engine,
                          bool force_fp32_output) {
  constexpr bool is_int8 =
      std::is_same<Tx, int8_t>::value || std::is_same<Tx, uint8_t>::value;
  if (!is_int8 || force_fp32_output) {
    GetPrimitiveFactory<Tx, Ty, float>(dev_ctx, ctx, x, y, mkldnn_engine)
        ->ExecuteMatMulPrimitive(x, y, output, ctx);
  } else {
    GetPrimitiveFactory<Tx, Ty, int8_t>(dev_ctx, ctx, x, y, mkldnn_engine)
        ->ExecuteMatMulPrimitive(x, y, output, ctx);
  }
}

template <typename Tx, typename Ty>
class MatMulDNNLOpKernel : public framework::OpKernel<Tx> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "MatMul MKL-DNN must use CPUPlace."));
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<LoDTensor>("X");
    auto y = ctx.Input<LoDTensor>("Y");
    auto output = ctx.Output<LoDTensor>("Out");

    bool transpose_x = ctx.Attr<bool>("transpoese_X");
    bool transpose_y = ctx.Attr<bool>("transpoese_Y");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    float alpha = ctx.Attr<float>("alpha");
    int head_n = ctx.Attr<int>("head_number");

    ExecuteMatMul<Tx, Ty>(dev_ctx, ctx, x, y, output, mkldnn_engine,
                          force_fp32_output);

    output->set_layout(DataLayout::kMKLDNN);
  }
};
}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(matmul, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kMATMUL_DNNL_FP32,
                                    ops::MatMulDNNLOpKernel<float, float>);

// REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(matmul, MKLDNN,
// ::paddle::platform::CPUPlace, U8,
// ops::kMATMUL_DNNL_FP32,
// ops::MatMulDNNLOpKernel<uint8_t, int8_t>);

// REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(matmul, MKLDNN,
// ::paddle::platform::CPUPlace, S8,
// ops::kMATMUL_DNNL_FP32,
// ops::MatMulDNNLOpKernel<int8_t, int8_t>);
