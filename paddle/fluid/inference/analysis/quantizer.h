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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_api.h"
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::NaiveExecutor;
using framework::Scope;
using framework::ProgramDesc;
using framework::LoDTensor;
using VarQuantMaxAndScale =
    std::map<std::string, std::pair<QuantMax, LoDTensor>>;

typedef std::function<bool(const std::vector<PaddleTensor>& inputs,
                           std::vector<PaddleTensor>* output_data,
                           int batch_size)>
    PredictorRun;

/* Class that performs quantization by running warm-up, calculating
 * scales and quantizing the graph by running quantize passes.
 */
class Quantizer final {
 public:
  explicit Quantizer(Scope* scope, std::shared_ptr<ProgramDesc> infer_program,
                     const std::shared_ptr<QuantizerConfig>& config,
                     const AnalysisConfig& aconfig, const Argument& aargument,
                     PredictorRun predictor_run)
      : scope_(scope),
        infer_program_(infer_program),
        config_(config),
        aconfig_(aconfig),
        aargument_(aargument),
        predictor_run_(predictor_run) {}

  // Execute full quantization procedure.
  bool Quantize();

 private:
  // Run single warmup iteration
  bool RunWarmup();
  // Gather data from variables and calculate scales for them.
  bool CalculateScales();
  // Calculate a scale for tensor based on ScaleAlgo rules.
  void CalculateSingleScale(const std::string& op_name,
                            const std::string& conn_name,
                            const std::string& var_name,
                            const LoDTensor* var_tensor);
  void PrepareArgument(Argument* arg);
  bool RunQuantizePasses();
  bool SaveModel();

 private:
  Scope* scope_;
  std::shared_ptr<ProgramDesc> infer_program_;
  const std::shared_ptr<QuantizerConfig>& config_;
  const AnalysisConfig& aconfig_;
  const Argument& aargument_;
  PredictorRun predictor_run_;

  // variable name -> data
  VarQuantMaxAndScale scales_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
