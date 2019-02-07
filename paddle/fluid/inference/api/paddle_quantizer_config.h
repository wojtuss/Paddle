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

// Algorithms for finding scale of quantized Tensors.
enum class ScaleAlgo {
  NONE,  // Do not compute scale (its done differently in passes)
  MAX,   // Find scale based on the maximum absolute value
  KL,    // Find scale based on KL Divergence
};

// The max value of a quantized integer.
enum class QuantMax : unsigned int {
  U8_MAX = 255,
  S8_MAX = 127,
};

struct QuantizerConfig {
  QuantizerConfig();

  /** Specify a quantization algorithm for a connection (input/output) of the
   * operator type.
   * @param op_type_name the operator's name.
   * @param conn_name name of the connection (input/output) of the operator.
   * @param alg the algorithm for computing scale.
   */
  void SetScaleAlgo(std::string op_type_name, std::string conn_name,
                    ScaleAlgo alg) {
    rules_[op_type_name][conn_name] = alg;
  }

  /** Get the quantization algorithm for a connection (input/output) of the
   * operator type.
   * @param op_type_name the operator's name.
   * @param conn_name name of the connection (input/output) of the operator.
   * @return the algorithm for computing scale.
   */
  ScaleAlgo scale_algo(const std::string& op_type_name,
                       const std::string& conn_name) const {
    return rules_.at(op_type_name).at(conn_name);
  }

  /** Set the batch of data to be used for warm-up iteration.
   * @param data batch of data.
   */
  void SetWarmupData(std::shared_ptr<std::vector<PaddleTensor>> data) {
    warmup_data_ = data;
  }

  /** Get the batch of data used for warm-up iteration.
   * @return batch of data.
   */
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data() {
    return warmup_data_;
  }

  void SetWamupBatchSize(int batch_size) { warmup_bs = batch_size; }

  int warmup_batch_size() const { return warmup_bs; }

  void SetQuantizeEnabledOpTypes(std::unordered_set<std::string> op_list) {
    quantize_enabled_op_types_ = op_list;
  }

  const std::unordered_set<std::string>& quantize_enabled_op_types() const {
    return quantize_enabled_op_types_;
  }

 protected:
  std::map<std::string, std::map<std::string, ScaleAlgo>> rules_;
  std::unordered_set<std::string> quantize_enabled_op_types_;
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data_;
  int warmup_bs{1};
};

}  // namespace paddle
