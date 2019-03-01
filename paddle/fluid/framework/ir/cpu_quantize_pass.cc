// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/cpu_quantize_pass.h"
#include <map>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"  // for QuantMax

namespace paddle {
namespace framework {
namespace ir {

using VarQuantMaxAndScale =
    std::map<std::string, std::pair<QuantMax, LoDTensor>>;

namespace {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

template <typename T>
void ScaleLoDTensor(LoDTensor* src, float scale) {
  auto* src_p = src->data<T>();
  for (int i = 0; i < src->numel(); ++i) {
    src_p[i] = static_cast<T>(std::round(src_p[i] * scale));
  }
}

template <typename T>
boost::optional<T> HasAttribute(const Node& op, const std::string& attr) {
  if (op.Op()->HasAttr(attr))
    return boost::get<T>(op.Op()->GetAttr(attr));
  else
    return boost::none;
}

}  // namespace

template <typename OutT>
void CPUQuantizePass::QuantizeInput(Graph* g, Node* op, Node* input,
                                    std::string input_name, std::string prefix,
                                    float scale, bool is_negative) const {
  // Create and initialize quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName(prefix + "quantize", "out"));
  quantize_out_desc.SetDataType(framework::ToDataType(typeid(OutT)));
  quantize_out_desc.SetShape(input->Var()->GetShape());
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);
  auto* quantize_out_tensor =
      param_scope()->Var(quantize_out_node->Name())->GetMutable<LoDTensor>();
  std::fill_n(quantize_out_tensor->mutable_data<OutT>(platform::CPUPlace()),
              quantize_out_tensor->numel(), 0);

  // create a quantize op node
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", scale);
  q_desc.SetAttr("is_negative_input", is_negative);
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // update op's input
  op->Op()->SetInput(input_name,
                     std::vector<std::string>({quantize_out_node->Name()}));

  // link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
}

template <typename InT>
void CPUQuantizePass::DequantizeOutput(Graph* g, Node* op, Node* output,
                                       std::string output_name,
                                       std::string prefix, float scale) const {
  // Create and initialize dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName(prefix + "dequantize", "in"));
  // dequantize_in_desc.SetDataType(proto::VarType::INT32);
  dequantize_in_desc.SetDataType(framework::ToDataType(typeid(InT)));
  dequantize_in_desc.SetShape(output->Var()->GetShape());
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);
  auto* dequantize_in_tensor =
      param_scope()->Var(dequantize_in_node->Name())->GetMutable<LoDTensor>();
  std::fill_n(dequantize_in_tensor->mutable_data<InT>(platform::CPUPlace()),
              dequantize_in_tensor->numel(), 0);

  // create a dequantize op node for output.
  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({output->Name()}));
  deq_desc.SetAttr("Scale", scale);
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  // update op's output
  op->Op()->SetOutput(output_name,
                      std::vector<std::string>({dequantize_in_node->Name()}));

  // link dequantize op
  UnlinkNodes(op, output);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, output);
}

void CPUQuantizePass::QuantizeResidualConn(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g,
    patterns::Conv conv_pattern, Node* conv_op, std::string prefix,
    PDPattern* base_pattern) const {
  GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                            conv_pattern);

  // Create and initialize quantize output variable
  VarDesc quantize_res_out_desc(
      patterns::PDNodeName(prefix + "q", "quantize_res_out"));
  quantize_res_out_desc.SetDataType(proto::VarType::INT32);
  auto* quantize_res_out_node = g->CreateVarNode(&quantize_res_out_desc);

  OpDesc q_res_conn_desc;
  q_res_conn_desc.SetType("quantize");
  q_res_conn_desc.SetInput(
      "Input", std::vector<std::string>({conv_residual_data->Name()}));
  q_res_conn_desc.SetOutput(
      "Output", std::vector<std::string>({quantize_res_out_node->Name()}));
  q_res_conn_desc.SetAttr("Scale", 1.0f);
  q_res_conn_desc.SetAttr("is_negative_input", true);
  auto quantize_op_res_conn = g->CreateOpNode(&q_res_conn_desc);

  conv_op->Op()->SetInput("ResidualData", std::vector<std::string>(
                                              {quantize_res_out_node->Name()}));

  UnlinkNodes(conv_residual_data, conv_op);
  IR_NODE_LINK_TO(conv_residual_data, quantize_op_res_conn);
  IR_NODE_LINK_TO(quantize_op_res_conn, quantize_res_out_node);
  IR_NODE_LINK_TO(quantize_res_out_node, conv_op);
}

void CPUQuantizePass::QuantizeInputOutput(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g,
    patterns::Conv conv_pattern, Node* conv_op, std::string prefix,
    std::pair<QuantMax, LoDTensor> conv_input_scales,
    std::pair<QuantMax, LoDTensor> conv_output_scales) const {
  GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
  GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);
  // Create and initialize quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName(prefix + "quantize", "out"));
  quantize_out_desc.SetDataType(proto::VarType::INT8);
  quantize_out_desc.SetShape(conv_input->Var()->GetShape());
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  auto* quantize_out_tensor =
      param_scope()->Var(quantize_out_node->Name())->GetMutable<LoDTensor>();
  // quantize_out_tensor->Resize(conv_input->Var()->GetShape());
  std::fill_n(quantize_out_tensor->mutable_data<int8_t>(platform::CPUPlace()),
              quantize_out_tensor->numel(), 0);

  // Create and initialize dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName(prefix + "dequantize", "in"));
  dequantize_in_desc.SetDataType(proto::VarType::INT32);
  dequantize_in_desc.SetShape(conv_output->Var()->GetShape());
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

  auto* dequantize_in_tensor =
      param_scope()->Var(dequantize_in_node->Name())->GetMutable<LoDTensor>();
  // quantize_out_tensor->Resize(conv_input->Var()->GetShape());
  std::fill_n(dequantize_in_tensor->mutable_data<int32_t>(platform::CPUPlace()),
              dequantize_in_tensor->numel(), 0);

  // create a quantize op node for input.
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({conv_input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));

  q_desc.SetAttr("Scale", conv_input_scales.second.data<float>()[0]);
  q_desc.SetAttr("is_negative_input",
                 conv_input_scales.first == QuantMax::S8_MAX);
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // create a dequantize op node for output.
  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({conv_output->Name()}));
  deq_desc.SetAttr("Scale", conv_output_scales.second.data<float>()[0]);
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  // update conv's inputs and outputs
  conv_op->Op()->SetInput(
      "Input", std::vector<std::string>({quantize_out_node->Name()}));
  conv_op->Op()->SetOutput(
      "Output", std::vector<std::string>({dequantize_in_node->Name()}));

  // link quantize op
  UnlinkNodes(conv_input, conv_op);
  IR_NODE_LINK_TO(conv_input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, conv_op);

  // link dequantize op
  UnlinkNodes(conv_op, conv_output);
  IR_NODE_LINK_TO(conv_op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, conv_output);
}

void CPUQuantizePass::QuantizePoolInputOutput(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g,
    patterns::Pool pool_pattern, Node* pool_op) const {
  GET_IR_NODE_FROM_SUBGRAPH(pool_input, pool_input, pool_pattern);
  GET_IR_NODE_FROM_SUBGRAPH(pool_output, pool_output, pool_pattern);

  // Create and initialize quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  quantize_out_desc.SetDataType(proto::VarType::INT8);
  quantize_out_desc.SetShape(pool_input->Var()->GetShape());
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  auto* quantize_out_tensor =
      param_scope()->Var(quantize_out_node->Name())->GetMutable<LoDTensor>();
  std::fill_n(quantize_out_tensor->mutable_data<int8_t>(platform::CPUPlace()),
              quantize_out_tensor->numel(), 0);

  // Create and initialize dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
  dequantize_in_desc.SetDataType(proto::VarType::INT8);
  dequantize_in_desc.SetShape(pool_output->Var()->GetShape());
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

  auto* dequantize_in_tensor =
      param_scope()->Var(dequantize_in_node->Name())->GetMutable<LoDTensor>();
  std::fill_n(dequantize_in_tensor->mutable_data<int8_t>(platform::CPUPlace()),
              dequantize_in_tensor->numel(), 0);

  // create a quantize op node for input.
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({pool_input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));

  using VarQuantMaxAndScale =
      std::map<std::string, std::pair<QuantMax, framework::LoDTensor>>;
  auto scales = Get<VarQuantMaxAndScale>("quant_var_scales");
  auto input_qmax_and_scale = scales[pool_input->Name()];
  q_desc.SetAttr("Scale", input_qmax_and_scale.second.data<float>()[0]);
  q_desc.SetAttr("is_negative_input",
                 input_qmax_and_scale.first == QuantMax::S8_MAX);
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // create a dequantize op node for output.
  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({pool_output->Name()}));
  deq_desc.SetAttr("Scale", input_qmax_and_scale.second.data<float>()[0]);
  auto dequantize_op = g->CreateOpNode(&deq_desc);

  // update op's inputs and outputs
  pool_op->Op()->SetInput(
      "X", std::vector<std::string>({quantize_out_node->Name()}));
  pool_op->Op()->SetOutput(
      "Out", std::vector<std::string>({dequantize_in_node->Name()}));

  // link quantize op
  UnlinkNodes(pool_input, pool_op);
  IR_NODE_LINK_TO(pool_input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, pool_op);

  // link dequantize op
  UnlinkNodes(pool_op, pool_output);
  IR_NODE_LINK_TO(pool_op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, pool_output);
}

void CPUQuantizePass::ScaleInput(Node* input, float scale) const {
  auto* input_tensor =
      param_scope()->Var(input->Name())->GetMutable<LoDTensor>();
  ScaleLoDTensor<float>(input_tensor, scale);
}

void CPUQuantizePass::QuantizeWeights(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g,
    patterns::Conv conv_pattern, Node* conv_op, std::string prefix,
    std::pair<QuantMax, LoDTensor> conv_filter_scales) const {
  GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
  auto* conv_filter_tensor =
      param_scope()->Var(conv_filter->Name())->GetMutable<LoDTensor>();
  float scale = conv_filter_scales.second.data<float>()[0];
  ScaleLoDTensor<float>(conv_filter_tensor, scale);
}

void CPUQuantizePass::QuantizeBias(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g,
    patterns::Conv conv_pattern, Node* conv_op, std::string prefix,
    std::pair<QuantMax, LoDTensor> conv_filter_scales,
    std::pair<QuantMax, LoDTensor> conv_input_scales) const {
  GET_IR_NODE_FROM_SUBGRAPH(conv_bias, conv_bias, conv_pattern);
  auto* conv_bias_tensor =
      param_scope()->Var(conv_bias->Name())->GetMutable<LoDTensor>();
  float scale = conv_filter_scales.second.data<float>()[0] *
                conv_input_scales.second.data<float>()[0];
  ScaleLoDTensor<float>(conv_bias_tensor, scale);
}

void CPUQuantizePass::QuantizeConv(Graph* graph, bool with_bias,
                                   bool with_res_conn) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Conv conv_pattern{pattern, name_scope_};
  conv_pattern(with_bias, with_res_conn);

  int quantize_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Conv2d with residual connection quantization";
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    auto* conv_op_desc = conv_op->Op();
    if (!conv_op_desc->HasAttr("use_quantizer") ||
        !boost::get<bool>(conv_op_desc->GetAttr("use_quantizer")))
      return;

    if (conv_op_desc->HasAttr("quantized") &&
        boost::get<bool>(conv_op_desc->GetAttr("quantized")))
      return;

    conv_op_desc->SetAttr("quantized", true);
    std::stringstream prefix_ss;
    if (with_bias) prefix_ss << "b_";
    if (with_res_conn) prefix_ss << "rc_";

    auto prefix = prefix_ss.str();

    auto scales = Get<VarQuantMaxAndScale>("quant_var_scales");
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);
    auto conv_input_scale = scales[conv_input->Name()].second.data<float>()[0];
    bool is_input_negative =
        scales[conv_input->Name()].first == QuantMax::S8_MAX;
    auto conv_filter_scale =
        scales[conv_filter->Name()].second.data<float>()[0];
    auto conv_output_scale = conv_input_scale * conv_filter_scale;

    QuantizeInput<int8_t>(g, conv_op, conv_input, "Input", prefix,
                          conv_input_scale, is_input_negative);
    DequantizeOutput<int32_t>(g, conv_op, conv_output, "Output", prefix,
                              conv_output_scale);

    ScaleInput(conv_filter, conv_filter_scale);

    if (with_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_bias, conv_bias, conv_pattern);
      ScaleInput(conv_bias, conv_output_scale);
    }

    if (with_res_conn) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                                conv_pattern);
      auto conv_res_conn_scale =
          scales[conv_residual_data->Name()].second.data<float>()[0];
      bool is_res_conn_negative =
          scales[conv_residual_data->Name()].first == QuantMax::S8_MAX;
      QuantizeInput<int32_t>(g, conv_op, conv_residual_data, "ResidualData",
                             prefix, conv_res_conn_scale, is_res_conn_negative);
    }

    ++quantize_conv_count;
  };

  gpd(graph, handler);
  std::stringstream msg_ss;
  msg_ss << "---  Quantized " << quantize_conv_count << " conv2d ops";
  if (with_bias) msg_ss << " with bias";
  if (with_res_conn) msg_ss << " and residual connection";
  msg_ss << "." << std::endl;
  std::cout << msg_ss.str();
  AddStatis(quantize_conv_count);
}

void CPUQuantizePass::QuantizePool(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Pool pool_pattern{pattern, name_scope_};
  pool_pattern();

  int quantize_pool_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Handle quantization of pool2d op";
    GET_IR_NODE_FROM_SUBGRAPH(pool_op, pool_op, pool_pattern);

    auto* pool_op_desc = pool_op->Op();
    if (!pool_op_desc->HasAttr("use_quantizer") ||
        !boost::get<bool>(pool_op_desc->GetAttr("use_quantizer")))
      return;

    if (pool_op_desc->HasAttr("quantized") &&
        boost::get<bool>(pool_op_desc->GetAttr("quantized")))
      return;

    pool_op_desc->SetAttr("quantized", true);

    std::string prefix{"aaa"};
    QuantizePoolInputOutput(subgraph, g, pool_pattern, pool_op);
    ++quantize_pool_count;
  };

  gpd(graph, handler);
  std::stringstream msg_ss;
  msg_ss << "---  Quantized " << quantize_pool_count << " pool2d ops."
         << std::endl;
  std::cout << msg_ss.str();
  AddStatis(quantize_pool_count);
}

std::unique_ptr<ir::Graph> CPUQuantizePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Quantizing the graph.";
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  PADDLE_ENFORCE(param_scope());

  QuantizeConv(graph.get(), true /* with_bias */, true /* with_res_conn */);
  QuantizeConv(graph.get(), true /* with_bias */);
  QuantizePool(graph.get());

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
