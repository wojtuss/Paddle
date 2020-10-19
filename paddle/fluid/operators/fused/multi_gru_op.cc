/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/multi_gru_op.h"
// #include "paddle/fluid/operators/fused/fusion_gru_op.h"
#include <cstring>  // for memcpy
#include <string>
#include <vector>
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/sequence2batch.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

void MultiGRUOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "multi_gru");
  OP_INOUT_CHECK(ctx->HasInputs("WeightX"), "Input", "WeightX", "multi_gru");
  OP_INOUT_CHECK(ctx->HasInputs("WeightH"), "Input", "WeightH", "multi_gru");
  OP_INOUT_CHECK(ctx->HasOutput("XX"), "Output", "XX", "multi_gru");
  OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "multi_gru");
  auto x_dims = ctx->GetInputDim("X");
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                        ? framework::flatten_to_2d(x_dims, 1)
                        : x_dims;
  PADDLE_ENFORCE_EQ(
      x_mat_dims.size(), 2,
      platform::errors::InvalidArgument("The size of input X dims should be 2, "
                                        "or 3 with second dimension equal to "
                                        "1, but now Input X dim is:[%s] ",
                                        x_dims));

  auto wx_dims = ctx->GetInputsDim("WeightX")[0];
  PADDLE_ENFORCE_EQ(wx_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightX) should be 2, but received "
                        "WeightX dim size is:%d, WeightX dim is:[%s] ",
                        wx_dims.size(), wx_dims));
  // PADDLE_ENFORCE_EQ(
  // wx_dims[0], x_mat_dims[1],
  // platform::errors::InvalidArgument(
  // "The first dimension of flattened WeightX"
  // "should equal to last dimension of flattened input X, but "
  // "received fattened WeightX dimension is:%d, flattened X dimension "
  // "is:%d",
  // wx_dims[0], x_mat_dims[1]));

  int frame_size = wx_dims[1] / 3;
  auto wh_dims = ctx->GetInputsDim("WeightH")[0];

  PADDLE_ENFORCE_EQ(wh_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightH) should be 2, but received "
                        "WeightH dim size is:%d, WeightH dim is:[%s]",
                        wh_dims.size(), wh_dims));
  PADDLE_ENFORCE_EQ(wh_dims[0], frame_size,
                    platform::errors::InvalidArgument(
                        "The first dimension of WeightH "
                        "should equal to frame_size, but received WeightH's "
                        "first dimension is: "
                        "%d, frame size is:%d",
                        wh_dims[0], frame_size));
  PADDLE_ENFORCE_EQ(wh_dims[1], 3 * frame_size,
                    platform::errors::InvalidArgument(
                        "The second dimension of Input(WeightH) "
                        "should equal to 3 * frame_size, but received WeightH "
                        "is:%d, frame size is:%d",
                        wh_dims[1], frame_size));

  if (ctx->HasInput("H0")) {
    auto h0_dims = ctx->GetInputDim("H0");
    PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                      platform::errors::InvalidArgument(
                          "The width of H0 must be equal to frame_size, but "
                          "receiced the width of H0 is:%d, frame size is:%d",
                          h0_dims[1], frame_size));
  }
  if (ctx->HasInputs("Bias")) {
    auto b_dims = ctx->GetInputsDim("Bias")[0];
    PADDLE_ENFORCE_EQ(b_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(Bias) should be 2, but received "
                          "Bias rank is:%d, Bias dim is:[%s]",
                          b_dims.size(), b_dims));
    PADDLE_ENFORCE_EQ(b_dims[0], 1,
                      platform::errors::InvalidArgument(
                          "The first dimension of Input(Bias) should be 1, but "
                          "received Bias first dim is:%d, Bias dim is:[%s]",
                          b_dims[0], b_dims));
    PADDLE_ENFORCE_EQ(b_dims[1], frame_size * 3,
                      platform::errors::InvalidArgument(
                          "The shape of Bias must be [1, frame_size * 3], but "
                          "received bias dim is:[%s], frame size is:%d",
                          b_dims, frame_size));
  }
  framework::DDim out_dims({x_mat_dims[0], 2 * frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->ShareLoD("X", "Hidden");
  int xx_width;
  if (ctx->Attrs().Get<bool>("use_seq")) {
    xx_width = wx_dims[1];
  } else {
    xx_width = x_mat_dims[1] > wx_dims[1] ? wx_dims[1] : x_mat_dims[1];
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedH0"), "Output", "ReorderedH0",
                   "multi_gru");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedInput"), "Output", "BatchedInput",
                   "multi_gru");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedOut"), "Output", "BatchedOut",
                   "multi_gru");
    ctx->SetOutputDim("BatchedInput", {x_mat_dims[0], wx_dims[1]});
    ctx->SetOutputDim("BatchedOut", out_dims);
  }
  ctx->SetOutputDim("XX", {x_mat_dims[0], xx_width});
  ctx->ShareLoD("X", "XX");
}

framework::OpKernelType MultiGRUOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kMKLDNN;
  framework::DataLayout layout = framework::DataLayout::kMKLDNN;

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(), layout,
      library);
}

void MultiGRUOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is an LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("H0",
           "(Tensor, optional) The initial hidden state is an optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size, D is the hidden size.")
      .AsDispensable();
  AddInput("WeightX",
           "(MultiTensor) The FC weight with shape (M x 3D),"
           "where M is the dim size of x, D is the hidden size. ");
  AddInput("WeightH",
           "(MultiTensor) (D x 3D) Same as GRUOp, where D is the hidden size. "
           "This weight is not exactly D x 3D as: {W_update, W_reset, W_state}"
           "Acutally they are D x 2D and D x D two part weights."
           "{W_update, W_reset; W_state}"
           "{D x (D + D); D x D}");
  AddInput("Bias",
           "(MultiTensor, optional) (1 x 3D)."
           "Almost same as GRUOp."
           "Note: if have FC bias it should be added on this bias.")
      .AsDispensable();
  AddInput(
      "Scale_weights",
      "(MultiTensor, optional) Scale_weights to be used for int8 weights data."
      "Only used with MKL-DNN INT8.")
      .AsDispensable();
  AddOutput("ReorderedH0", "(Tensor) (N x D), which N is the min-batch size.")
      .AsIntermediate();
  AddOutput("XX",
            "(LoDTensor) the result after X * WeightX (size is T x 3D)"
            " or batched_X (size is T x M), this will be automatically chosen,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size, M is the dim size of x input.")
      .AsIntermediate();
  AddOutput("BatchedInput",
            "(LoDTensor) This is the batched result of input X"
            "or the batched result after fc, shape (T x 3D)")
      .AsIntermediate();
  AddOutput("BatchedOut", "(LoDTensor) (T X D) save batched hidden.")
      .AsIntermediate();
  AddOutput("Hidden", "(LoDTensor) (T x D) Same as GRUOp");
  AddAttr<std::string>("activation",
                       "(string, default tanh) "
                       "The activation type used for output candidate {h}_t.")
      .SetDefault("tanh");
  AddAttr<std::string>(
      "gate_activation",
      "(string, default sigmoid) "
      "The activation type used in update gate and reset gate.")
      .SetDefault("sigmoid");
  AddAttr<int>("layers",
               "(int, default: 1) "
               "Number of stacked GRU layers.")
      .SetDefault(1);
  AddAttr<bool>("use_seq",
                "(bool, default: True) "
                "whether to use seq mode to compute GRU.")
      .SetDefault(true);
  AddAttr<bool>("origin_mode",
                "bool"
                "use origin mode in article https://arxiv.org/abs/1412.3555")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  // AddAttr<std::vector<float>>("Scale_data",
  AddAttr<float>("Scale_data",
                 "Scales to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault({1.f});
  // AddAttr<std::vector<float>>("Shift_data",
  AddAttr<float>("Shift_data",
                 "Shifts to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault({0.f});
  AddAttr<bool>("force_fp32_output",
                "(bool, default: false) Force INT8 kernel output FP32, only "
                "used in MKL-DNN INT8")
      .SetDefault(false);
  AddComment(R"DOC(
The Fusion complete GRU Operator.
This operator fuse the fully-connected operator into GRU, 
more details can refer to GRU op.
)DOC");
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multi_gru, ops::MultiGRUOp, ops::MultiGRUOpMaker);
