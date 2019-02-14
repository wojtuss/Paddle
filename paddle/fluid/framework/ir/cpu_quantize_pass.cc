// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace framework {
namespace ir {

namespace {

enum class OpTypes { conv2d, pool2d };

std::map<std::string, OpTypes> string2OpType{
    std::make_pair("conv2d", OpTypes::conv2d),
    std::make_pair("pool2d", OpTypes::pool2d)};

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

void QuantizeLoDTensor(const LoDTensor& src, LoDTensor* dst) {
  // Quantize
}

}  // namespace

std::unique_ptr<ir::Graph> CPUQuantizePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Quantizes the graph.";
  std::cout << "--- This is cpu quantize pass. ---" << std::endl;
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope_};
  conv_pattern();

  int quantize_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Conv2d quantization";
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    auto* conv_op_desc = conv_op->Op();
    if (!conv_op_desc->HasAttr("use_quantizer") ||
        !boost::get<bool>(conv_op_desc->GetAttr("use_quantizer")))
      return;

    // insert quantize op

    // Create eltwise_y (conv bias) variable
    VarDesc quantize_out_desc(
        patterns::PDNodeName(name_scope_, "quantize_out"));
    auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

    // create a quantize op node.
    OpDesc desc;
    desc.SetType("quantize");
    desc.SetInput("Input", std::vector<std::string>({conv_output->Name()}));
    desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
    desc.SetAttr("Scale", 1.0f);
    desc.SetAttr("is_negative_input", true);
    auto quantize_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

    // conv_op_desc->SetInput(
    // "Input", std::vector<std::string>({quantize_out_node->Name()}));
    conv_op_desc->SetInput("Input", std::vector<std::string>({"aaa"}));

    IR_NODE_LINK_TO(conv_input, quantize_op);         // Input
    IR_NODE_LINK_TO(quantize_op, quantize_out_node);  // Output
    IR_NODE_LINK_TO(quantize_out_node, conv_op);      // Output
    UnlinkNodes(conv_input, conv_op);

    // insert dequantize op
    // quantize weights
    // quantize bias
    // update op?

    ++quantize_conv_count;
  };

  gpd(graph.get(), handler);
  AddStatis(quantize_conv_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
