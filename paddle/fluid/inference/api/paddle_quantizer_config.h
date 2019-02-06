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
#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle_api.h"  // NOLINT

namespace paddle {

enum QuantizeAlgorithm {
  none,
  minmax,
  KL,
};

struct QuantizerConfig {
  QuantizerConfig();

  void SetScaleAlgo(std::string op_name, std::string var_name,
                    QuantizeAlgorithm alg) {
    rules_[op_name][var_name] = alg;
  }

  /** Specify the operator type list to use INT8 kernel.
   * @param op_list the operator type list.
   */
  void SetOps(std::unordered_set<std::string> op_list) {
    quantize_enabled_op_types_ = op_list;
  }

  void SetWarmupData(std::shared_ptr<std::vector<PaddleTensor>> data) {
    warmup_data_ = data;
  }

  std::shared_ptr<std::vector<PaddleTensor>> GetWarmupData() {
    return warmup_data_;
  }

  int GetWarmupBatchSize() { return warmup_bs; }
  const std::unordered_set<std::string>& GetQuantizeEnabledOpTypes() {
    return quantize_enabled_op_types_;
  }

  QuantizeAlgorithm GetScaleAlgo(const std::string& op_name,
                                 const std::string& conn_name) {
    return rules_[op_name][conn_name];
  }

  friend struct AnalysisConfig;

 protected:
  std::map<std::string, std::map<std::string, QuantizeAlgorithm>> rules_;
  std::unordered_set<std::string> quantize_enabled_op_types_;
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data_;
  int warmup_bs{0};
};

}  // namespace paddle
