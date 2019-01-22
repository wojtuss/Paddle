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

/*
 * This file defines IRPassManager, it helps control the passes in IR. Inference
 * phrase will load the model program and parameters from disk, that is quite
 * different from the training phase.
 * This manager will control the Passes and make the passes in IR work smoothly
 * for inference.
 */

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::NaiveExecutor;
// using EigenVectorArrayMap = Eigen::Map<Eigen::Array<float, Eigen::Dynamic,
// 1>>;
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
using framework::Scope;
using framework::ProgramDesc;
using framework::LoDTensor;
using contrib::QuantizerConfig;

typedef std::function<bool(const std::vector<PaddleTensor>& inputs,
                           std::vector<PaddleTensor>* output_data,
                           int batch_size)>
    PredictorRun;

class Quantizer final {
 public:
  explicit Quantizer(Scope* scope,
                     const std::shared_ptr<ProgramDesc>& infer_program,
                     const std::shared_ptr<QuantizerConfig>& config,
                     PredictorRun predictor_run)
      : scope_(scope),
        infer_program_(infer_program),
        config_(config),
        predictor_run_(predictor_run) {}

  bool Quantize();

 private:
  bool RunWarmup();
  bool GatherData();
  void CalculateScales(const std::string& op_name, const std::string& conn_name,
                       const std::string& var_name, const LoDTensor& var_tensor,
                       float var_max_range);
  bool RunQuantizePass();
  bool RunOptimizePass();
  bool SaveModel();
  float GetOptimalScalingFactor(ConstEigenVectorArrayMap eigen_data_vector,
                                int num_quantized_bins = 255);
  float GetMaxScalingFactor(ConstEigenVectorArrayMap activation_blob,
                            float var_max_range);
  std::pair<std::vector<int>, float> Histogram(
      ConstEigenVectorArrayMap activation_blob, float min_val, float max_val,
      int num_bins = 2048);
  std::vector<int> ExpandQuantizedBins(std::vector<int> quantized_bins,
                                       std::vector<int> reference_bins);
  float safe_entropy(std::vector<int> reference_distr_P, int P_sum,
                     std::vector<int> candidate_distr_Q, int Q_sum);

 private:
  Scope* scope_;
  const std::shared_ptr<ProgramDesc>& infer_program_;
  const std::shared_ptr<QuantizerConfig>& config_;
  PredictorRun predictor_run_;

  // variable name -> data
  std::map<std::string, framework::LoDTensor> scales_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
