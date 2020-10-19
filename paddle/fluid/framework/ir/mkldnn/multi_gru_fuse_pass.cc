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

#include "paddle/fluid/framework/ir/mkldnn/multi_gru_fuse_pass.h"
#include <limits>
#include <sstream>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using EigenVectorArrayMap = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>>;
using string::PrettyLogDetail;

namespace {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

std::vector<std::string> join_inputs(Node* op1, Node* op2,
                                     std::string input_name) {
  auto in1 = op1->Op()->Input(input_name);
  auto in2 = op2->Op()->Input(input_name);
  in1.insert(in1.end(), in2.begin(), in2.end());
  return in1;
}

}  // namespace

void MultiGRUFusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Fusing two concatenated multi_gru ops.";
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument cannot be NULL."));
  FusePassBase::Init(name_scope_, graph);
  PADDLE_ENFORCE_NOT_NULL(param_scope(), platform::errors::InvalidArgument(
                                             "Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  patterns::TwoFusionGruConcat pattern{gpd.mutable_pattern(), name_scope_};
  pattern();

  int fused_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(x, x, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru1, gru1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru2, gru2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh1, wh1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh2, wh2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx1, wx1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx2, wx2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b1, b1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b2, b2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(h1, h1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(h2, h2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat, concat, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out, out, pattern);

    auto wx = join_inputs(gru1, gru2, "WeightX");
    auto wh = join_inputs(gru1, gru2, "WeightH");
    auto b = join_inputs(gru1, gru2, "Bias");

    OpDesc multi_gru_desc;
    multi_gru_desc.SetType("multi_gru");
    multi_gru_desc.SetInput("X", std::vector<std::string>({x->Name()}));
    multi_gru_desc.SetInput("WeightX", wx);
    multi_gru_desc.SetInput("WeightH", wh);
    multi_gru_desc.SetInput("Bias", b);
    multi_gru_desc.SetOutput("Hidden", std::vector<std::string>({out->Name()}));
    for (auto& attr : gru1->Op()->GetAttrMap()) {
      multi_gru_desc.SetAttr(attr.first, attr.second);
    }
    multi_gru_desc.SetAttr("layers", 1);
    auto multi_gru =
        g->CreateOpNode(&multi_gru_desc);  // OpDesc will be copied.

    for (auto out_name : {"BatchedInput", "BatchedOut", "ReorderedH0", "XX"}) {
      auto var_name = name_scope_ + "/" + out_name;
      multi_gru->Op()->SetOutput(out_name,
                                 std::vector<std::string>({var_name}));
      VarDesc var_desc(var_name);
      var_desc.SetPersistable(false);
      auto* out_node = graph->CreateVarNode(&var_desc);
      IR_NODE_LINK_TO(multi_gru, out_node);
    }

    IR_NODE_LINK_TO(x, multi_gru);
    IR_NODE_LINK_TO(b1, multi_gru);
    IR_NODE_LINK_TO(b2, multi_gru);
    IR_NODE_LINK_TO(wh1, multi_gru);
    IR_NODE_LINK_TO(wh2, multi_gru);
    IR_NODE_LINK_TO(wx1, multi_gru);
    IR_NODE_LINK_TO(wx2, multi_gru);
    IR_NODE_LINK_TO(multi_gru, out);
    GraphSafeRemoveNodes(graph, {gru1, gru2, h1, h2, concat});

    ++fused_count;
  };
  gpd(graph, handler);
  AddStatis(fused_count);

  PrettyLogDetail("---    fused %d pairs of concatenated multi_gru ops",
                  fused_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_gru_fuse_pass, paddle::framework::ir::MultiGRUFusePass);
