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

#include <string>
#include "paddle/fluid/framework/op_registry.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

class GRUFusedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of GRUFusedOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("WeightX"),
                   "Input(%s) of GRUFusedOp should not be null.", "WeightX");
    PADDLE_ENFORCE(ctx->HasInput("WeightH"),
                   "Input(%s) of GRUFusedOp should not be null.", "WeightH");
    PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                   "Output(%s) of GRUFusedOp should not be null.", "Hidden");
    PADDLE_ENFORCE(ctx->HasAttr("activation"),
		   "Attr(%s) of GRUFusedOp should not be null.", "activation");
    PADDLE_ENFORCE(ctx->HasAttr("gate_activation"),
	           "Attr(%s) of GRUFusedOp should not be null.",
		   "gate_activation");
    PADDLE_ENFORCE(ctx->HasAttr("stack_level"),
		   "Attr(%s) of GRUFusedOp should not be null.", "stack_level");
    PADDLE_ENFORCE(ctx->HasAttr("is_bidirection"),
		   "Attr(%s) of GRUFusedOp should not be null.",
		   "is_bidirection");
    PADDLE_ENFORCE(ctx->HasAttr("is_reverse"),
		   "Attr(%s) of GRUFusedOp should not be null.", "is_reverse");
    
    auto input_dims = ctx->GetInputDim("Input");
    auto weight_h_dims = ctx->GetInputDim("WeightH");
    auto weight_x_dims = ctx->GetInputDim("WeightX");

    bool is_rev = ctx->Attrs().Get<bool>("is_reverse");
    bool is_bidir = ctx->Attrs().Get<bool>("is_bidirection");
    int D = is_bidir ? 2 : 1;
    int L = ctx->Attrs().Get<int>("stack_level");

    PADDLE_ENFORCE(!(is_rev && is_bidir),
		   "Attr(%s) and Attr(%s) should not be set both true",
		   "is_reverse", "is_bidirection");
    PADDLE_ENFORCE_GT(L, 0,
		   "Attr(%s) of GRUFusedOP should be > 0", "stack_level");

    int T = input_dims[0];       // Total time steps in batch
    int I = input_dims[1];       // Input feature size
    int C = weight_h_dims[0];    // Hidden state size
    int G = 3;                   // Gate of GRU is 3
    
    PADDLE_ENFORCE_EQ(weight_h_dims[1], L * D * G * C,
        "The shape of WeightH matrix must be [C, L * D * G * C].");
    PADDLE_ENFORCE_EQ(weight_x_dims[0], I,
        "The shape of WeightX matrix must be [I, L * D * G * C].");
    PADDLE_ENFORCE_EQ(weight_x_dims[1], L * D * 3 * C,
        "The shape of WeightX matrix must be [I, L * D * G * C].");

    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(h0_dims[1], L * D * C,
                        "The width of H0 must be equal to L * D * C.");
    }
    
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      PADDLE_ENFORCE_EQ(bias_dims[0], 1,
                        "The shape of Bias must be [1, L * D * G * C].");
      PADDLE_ENFORCE_EQ(bias_dims[1], L * D * 3 *C,
                        "The shape of Bias must be [1, L * D * G * C].");
    }
    ctx->SetOutputDim("Hidden", {T, D * C});
    ctx->ShareLoD("Input", "Hidden");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library{framework::LibraryType::kPlain};
    std::string data_format = ctx.Attr<std::string>("data_format");
    framework::DataLayout layout = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

  return framework::OpKernelType(
	    framework::ToDataType(ctx.Input<Tensor>("Input")->type()),
	    ctx.GetPlace(), layout, library);
  }
};

class GRUFusedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(LoDTensor) A variable-time length input sequence. The "
	"underlying tensor in this LoDTensor is a matrix with "
	"shape (T x I) where T is the total time steps in this "
	"mini-batch, I is the input feature size.");
    AddInput(
	"H0",
        "(Tensor, optional) The initial hidden state is an optional "
        "input. This is a tensor with shape (N x L*D*C), where N is "
        "the batch size, L is the stacked layers, D is 2 for "
	"bidirectional or 1 otherwise, C is the hidden state size.")
        .AsDispensable();
    AddInput(
        "WeightH",
        "(Tensor) The learnable hidden-hidden weight matrix with shape "
        "(C x L*D*3*C), where C is the hidden size. L is the stacked "
	"layers, D is 2 for bidirectional or 1 otherwies, 3 here means "
	"three gates in GRU.");
    AddInput(
	"WeightX",
	"(Tensor) The learnable input-hidden weight matrix with shape "
	"(I x L*D*3*C), where I is the input feature size and C is the "
	"hidden size, L is the stacked layers, D is 2 for bidirectional "
	"or 1 otherwise, 3 here means three gates in GRU.");
    AddInput(
	"Bias",
        "(Tensor, optional) Bias vector with shape (1 x L*D*3*C), where"
	"L is the stacked layers, D is 2 for bidirectional or 1 otherwise,"
	"3 here means three gates in GRU, C is the hidden state size.")
        .AsDispensable();
    AddOutput(
        "Hidden",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D*C) where T is the "
	"total time steps in the mini-batch, D is 2 for bidirectional or "
	"1 otherwise, C is the hidden state size.");
    AddAttr<std::string>("activation",
                         "(string, default tanh) "
                         "The activation type used for output candidate {h}_t.")
        .SetDefault("tanh");
    AddAttr<std::string>(
        "gate_activation",
        "(string, default sigmoid) "
        "The activation type used in update gate and reset gate.")
        .SetDefault("sigmoid");
    AddAttr<bool>("is_reverse",
                  "(bool, defalut: False) "
                  "whether to compute reversed GRU.")
        .SetDefault(false);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) "
		  "Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<int>("stack_level",
		 "(int, default: 1) "
		 "how many layers of GRU will be stacked, i.e. L")
      .SetDefault(1);
    AddAttr<bool>("is_bidirection",
		  "(bool, default False) "
		  "whether bi-direction GRU, i.e. D")
      .SetDefault(false);
    AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("AnyLayout");  
    AddComment(R"DOC(
Fused GRU Operator implements calculations of the complete GRU as following:

$$
update\_gate: u_t = actGate(WX_u * X + WU_u * h_{t-1} + b_u) \\
reset\_gate: r_t = actGate(WX_r * X + WU_r * h_{t-1} + b_r)  \\
output\_candidate: {h}_t = actNode(WX_c + WU_c * dot(r_t, h_{t-1}) + b_c) \\
output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
$$
)DOC");
  }
};

template <typename DeviceContext, typename T>
class GRUFusedDummyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(1==0, "This is a dummy kernel.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(gru_fused, ops::GRUFusedOp, ops::GRUFusedOpMaker)
REGISTER_OP_CPU_KERNEL(
    gru_fused, ops::GRUFusedDummyKernel<paddle::platform::CPUDeviceContext, float>)
