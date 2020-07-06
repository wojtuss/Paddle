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

#include "paddle/fluid/framework/ir/conv_bif_bn_fuse_pass.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

namespace {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

}  // namespace

Node* ConvBifBNFusePass::FindOpInputNodeByName(
    const Graph* graph, const Node* op, const std::string& input_name) const {
  auto input_var_names = op->Op()->Input(input_name);
  PADDLE_ENFORCE_EQ(input_var_names.size(), 1,
                    platform::errors::InvalidArgument(
                        "The %s input has more than 1 name (%d).", input_name,
                        input_var_names.size()));
  auto input_var_name = input_var_names[0];
  auto node_iter = std::find_if(op->inputs.begin(), op->inputs.end(),
                                [&input_var_name](const Node* node) -> bool {
                                  return node->Name() == input_var_name;
                                });
  PADDLE_ENFORCE_NE(
      node_iter, op->inputs.end(),
      platform::errors::NotFound("A node named %s not found in the graph.",
                                 input_var_name));
  return *node_iter;
}

Node* ConvBifBNFusePass::CopyPersistableNode(Graph* graph,
                                             const Node* node) const {
  PADDLE_ENFORCE_EQ(node->IsVar(), true,
                    platform::errors::InvalidArgument(
                        "The node argument must be a variable node."));
  auto node_tensor = param_scope()->FindVar(node->Name())->Get<LoDTensor>();
  VarDesc node_copy_desc(
      patterns::PDNodeName(name_scope_, node->Name() + "_copy"));
  node_copy_desc.SetShape(vectorize(node_tensor.dims()));
  node_copy_desc.SetDataType(node_tensor.type());
  node_copy_desc.SetLoDLevel(node->Var()->GetLoDLevel());
  node_copy_desc.SetPersistable(true);
  auto* node_copy = graph->CreateVarNode(&node_copy_desc);
  auto* node_copy_tensor =
      param_scope()->Var(node_copy->Name())->GetMutable<LoDTensor>();
  TensorCopy(node_tensor, platform::CPUPlace(), node_copy_tensor);
  return node_copy;
}

Node* ConvBifBNFusePass::CopyActivationNode(Graph* graph,
                                            const Node* node) const {
  PADDLE_ENFORCE_EQ(node->IsVar(), true,
                    platform::errors::InvalidArgument(
                        "The node argument must be a variable node."));
  VarDesc node_copy_desc(
      patterns::PDNodeName(name_scope_, node->Name() + "_copy"));
  node_copy_desc.SetPersistable(false);
  return graph->CreateVarNode(&node_copy_desc);
}

Node* ConvBifBNFusePass::CopyOpNode(Graph* graph, const Node* op) const {
  PADDLE_ENFORCE_EQ(op->IsOp(), true,
                    platform::errors::InvalidArgument(
                        "The node argument must be a operator node."));
  OpDesc op_copy_desc;
  op_copy_desc.CopyFrom(*op->Op());
  op_copy_desc.Flush();
  return graph->CreateOpNode(&op_copy_desc);
}

void ConvBifBNFusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Fusing conv+(bn,any,...).";
  PADDLE_ENFORCE(graph);
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  GraphPatternDetector gpd;
  auto mutable_pattern = gpd.mutable_pattern();
  patterns::ConvBifBN pattern{mutable_pattern, name_scope_};
  pattern();

  int fused_pattern_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle conv2d+(bn,any,...) fuse";

    if (fused_pattern_count > 0) return;

    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn, bn, pattern);

    auto* conv_copy_filter = CopyPersistableNode(g, conv_filter);

    // Make a copy of the bias, if present
    bool has_bias =
        conv->Op()->HasInput("Bias") && conv->Op()->Input("Bias").size() > 0;
    Node* conv_copy_bias = nullptr;
    if (has_bias) {
      auto* conv_bias = FindOpInputNodeByName(g, conv, "Bias");
      conv_copy_bias = CopyPersistableNode(g, conv_bias);
    }

    // Find a node with residual data, if present
    bool has_residual_data = conv->Op()->HasInput("ResidualData") &&
                             conv->Op()->Input("ResidualData").size() > 0;
    Node* conv_residual_data = nullptr;
    if (has_residual_data) {
      conv_residual_data = FindOpInputNodeByName(g, conv, "ResidualData");
    }

    // Create an output for the copied conv op
    auto* conv_copy_output = CopyActivationNode(g, conv_output);

    // Make a copy of the conv op node
    auto* conv_copy = CopyOpNode(g, conv);

    // Update the inputs and output
    conv_copy->Op()->SetInput(
        "Filter", std::vector<std::string>({conv_copy_filter->Name()}));
    if (has_bias) {
      conv_copy->Op()->SetInput(
          "Bias", std::vector<std::string>({conv_copy_bias->Name()}));
    }
    conv_copy->Op()->SetOutput(
        "Output", std::vector<std::string>({conv_copy_output->Name()}));

    bn->Op()->SetInput("X",
                       std::vector<std::string>({conv_copy_output->Name()}));

    IR_NODE_LINK_TO(conv_input, conv_copy);
    IR_NODE_LINK_TO(conv_copy_filter, conv_copy);
    if (has_bias) IR_NODE_LINK_TO(conv_copy_bias, conv_copy);
    if (has_residual_data) IR_NODE_LINK_TO(conv_residual_data, conv_copy);
    IR_NODE_LINK_TO(conv_copy, conv_copy_output);
    IR_NODE_LINK_TO(conv_copy_output, bn);

    UnlinkNodes(conv_output, bn);
    fused_pattern_count++;
  };

  gpd(graph, handler);
  AddStatis(fused_pattern_count);
  PrettyLogDetail("---    fused %d ConvBifBN patterns", fused_pattern_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_bif_bn_fuse_pass, paddle::framework::ir::ConvBifBNFusePass);
