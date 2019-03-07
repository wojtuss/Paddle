// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file eint8_outcept in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eint8_outpress or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/cpu_quantize_squash_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void CPUQuantizeSquashPass::SingleBranch(Graph* graph) const{
  GraphPatternDetector gpd;
  auto* int8_out = gpd.mutable_pattern()
                ->NewNode("squash_pass/int8_out")
                ->AsInput()
                ->assert_is_op_input("dequantize", "Input");

  patterns::DequantQuantRM squash_pattern(gpd.mutable_pattern(),"squash_pass");
  squash_pattern(int8_out);

  int found_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle cpu quantize squash pass";
    GET_IR_NODE_FROM_SUBGRAPH(dequant, dequantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant, quantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);

    auto* dequant_op_desc = dequant->Op();
    auto* quant_op_desc = quant->Op();
    float dequant_scale = boost::get<float>(dequant_op_desc->GetAttr("Scale"));
    float quant_scale = boost::get<float>(quant_op_desc->GetAttr("Scale"));

    if (dequant_scale == quant_scale){
       //remove the dequantize and quantize op
       GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out, quant_out});

       PADDLE_ENFORCE(subgraph.count(int8_out));
       IR_NODE_LINK_TO(subgraph.at(int8_out), next_op);

       found_squash_count++;
    }else{
       //Create an requantize Node
       OpDesc desc;
       std::string squash_int8_out_in = subgraph.at(int8_out)->Name();
       std::string squash_out = quant_out->Name();
       desc.SetInput("Input", std::vector<std::string>({squash_int8_out_in}));
       desc.SetOutput("Output", std::vector<std::string>({squash_out}));
       desc.SetAttr("Scale_dequant", dequant->Op()->GetAttr("Scale"));
       desc.SetAttr("Scale_quant", quant->Op()->GetAttr("Scale"));
       desc.SetAttr("is_negative_input", quant->Op()->GetAttr("is_negative_input"));
       desc.SetType("requantize");

       auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
       GraphSafeRemoveNodes(graph, {dequant, quant,  dequant_out});

       PADDLE_ENFORCE(subgraph.count(int8_out));
       IR_NODE_LINK_TO(subgraph.at(int8_out), requant_node);
       IR_NODE_LINK_TO(requant_node, quant_out);

       found_squash_count++;
    }
};
  gpd(graph, handler);
  AddStatis(found_squash_count);
}


std::unique_ptr<ir::Graph> CPUQuantizeSquashPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("cpu_quantize_squash_pass", graph.get());
  
  SingleBranch(graph.get()); 

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
  
  
