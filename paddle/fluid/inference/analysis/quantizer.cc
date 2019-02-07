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

#include "paddle/fluid/inference/analysis/quantizer.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <utility>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace inference {
namespace analysis {

using platform::CPUPlace;
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;

namespace {

std::pair<QuantMax, LoDTensor> GetMaxScalingFactor(LoDTensor var_tensor) {
  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  float min_val = eigen_tensor.minCoeff();
  bool is_positive = min_val >= 0.0f;
  auto quant_max = is_positive ? QuantMax::U8_MAX : QuantMax::S8_MAX;
  if (!(quant_max == QuantMax::U8_MAX || quant_max == QuantMax::S8_MAX))
    throw std::runtime_error(
        "Quantizer: Only 8 bit quantization is supported now.");

  LoDTensor scale_tensor;
  scale_tensor.Resize({1});
  auto* scale_ptr = scale_tensor.mutable_data<float>(CPUPlace());

  float max_abs = eigen_tensor.abs().maxCoeff();
  scale_ptr[0] = static_cast<float>(static_cast<unsigned>(quant_max) / max_abs);

  return std::make_pair(quant_max, scale_tensor);
}

// Returns histogram and bin width
std::pair<std::vector<int>, float> Histogram(
    ConstEigenVectorArrayMap eigen_tensor, float min_val, float max_val,
    int num_bins = 2048) {
  auto bin_width = (max_val - min_val) / num_bins;
  std::vector<int> hist(num_bins);
  // TODO(sfraczek): Try #pragma omp parallel for
  for (int i = 0; i < eigen_tensor.size(); i++) {
    int bin = static_cast<int>(floor((eigen_tensor[i] - min_val) / bin_width));
    hist[bin] = eigen_tensor[i];
  }
  // TODO(sfraczek): Try the below
  // auto bin_blob = (eigen_tensor - min_val) / bin_width
  // for int (i = 0; i < bin_blob.size(); i++)
  //   hist[bin_blob[i]]=eigen_tensor[i];

  return std::make_pair(std::move(hist), std::move(bin_width));
}

std::vector<int> ExpandQuantizedBins(std::vector<int> quantized_bins,
                                     std::vector<int> reference_bins) {
  std::vector<int> expanded_quantized_bins(reference_bins.size(), 0);
  int num_merged_bins = reference_bins.size() / quantized_bins.size();
  int j_start = 0;
  int j_end = num_merged_bins;
  for (size_t idx = 0; idx < quantized_bins.size(); idx++) {
    // auto zero_count = reference_bins[j_start:j_end].count(0);
    int zero_count =
        std::count(&reference_bins[j_start], &reference_bins[j_end], 0);
    num_merged_bins = j_end - j_start;
    int avg_bin_ele;
    if (zero_count == num_merged_bins) {
      avg_bin_ele = 0;
    } else {
      avg_bin_ele = quantized_bins[idx] / (num_merged_bins - zero_count + 0.0);
    }
    for (int idx1 = j_start; idx1 < j_end; idx1++) {
      expanded_quantized_bins[idx1] =
          (reference_bins[idx1] == 0) ? 0 : avg_bin_ele;
    }
    j_start += num_merged_bins;
    j_end += num_merged_bins;
    if ((idx + 1) == quantized_bins.size() - 1) {
      j_end = reference_bins.size();
    }
  }
  return expanded_quantized_bins;
}

// Calculate the entropy.
float SafeEntropy(std::vector<int> reference_distr_P, int P_sum,
                  std::vector<int> candidate_distr_Q, int Q_sum) {
  PADDLE_ENFORCE_EQ(reference_distr_P.size(), candidate_distr_Q.size());
  float tmp_sum1 = 0;
  float tmp_sum2 = 0;
  for (size_t idx = 0; idx < reference_distr_P.size(); idx++) {
    int p_idx = reference_distr_P[idx];
    int q_idx = candidate_distr_Q[idx];
    if (p_idx == 0) {
      tmp_sum1 += 0;
      tmp_sum2 += 0;
    } else {
      if (q_idx == 0) {
        throw std::runtime_error("Quantizer: Fatal error!, idx = " +
                                 std::to_string(idx) + " qindex = 0! p_idx = " +
                                 std::to_string(p_idx));
      }
      tmp_sum1 += p_idx * (log(Q_sum * p_idx));
      tmp_sum2 += p_idx * (log(P_sum * q_idx));
    }
  }
  return (tmp_sum1 - tmp_sum2) / P_sum;
}

// Using the KL-divergence method get the most precise scaling factor.
std::pair<QuantMax, LoDTensor> GetKLScalingFactor(LoDTensor var_tensor) {
  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  float max_val = eigen_tensor.maxCoeff();
  float min_val = eigen_tensor.minCoeff();
  bool is_positive = min_val >= 0.0f;
  auto quant_max = is_positive ? QuantMax::U8_MAX : QuantMax::S8_MAX;

  int num_quantized_bins;
  if (quant_max == QuantMax::U8_MAX || quant_max == QuantMax::S8_MAX)
    num_quantized_bins = 255;
  else
    throw std::runtime_error(
        "Quantizer: Only 8 bit quantization is supported now.");

  std::vector<int> hist;
  float bin_width;
  int starting_iter;
  int ending_iter;
  if (is_positive) {
    std::tie(hist, bin_width) = Histogram(eigen_tensor, min_val, max_val, 2048);
    ending_iter = 2047;
    starting_iter = static_cast<int>(ending_iter * 0.7);
  } else {
    float th = std::max(abs(max_val), abs(min_val));
    std::tie(hist, bin_width) = Histogram(eigen_tensor, -th, th, 2048);
    starting_iter = 0;
    ending_iter = 2047;
    if (abs(max_val) > abs(min_val)) {
      while (starting_iter < ending_iter) {
        if (hist[starting_iter] == 0) {
          starting_iter += 1;
          continue;
        } else {
          break;
        }
      }
      starting_iter += static_cast<int>((ending_iter - starting_iter) * 0.6);
    } else {
      while (ending_iter > 0) {
        if (hist[ending_iter] == 0) {
          ending_iter -= 1;
          continue;
        } else {
          break;
        }
      }
      starting_iter = static_cast<int>(0.6 * ending_iter);
    }
  }
  auto P_sum = eigen_tensor.size();
  int min_kl_divergence = 0;
  int min_kl_index = 0;
  bool kl_inited = false;
  for (int i = starting_iter; i <= ending_iter; i++) {
    std::vector<int> reference_distr_P(&hist[0], &hist[i]);
    auto outliers_count = std::accumulate(&hist[i], &hist[2048], 0);
    if (reference_distr_P[i - 1] == 0) {
      continue;
    }
    reference_distr_P[i - 1] += outliers_count;
    auto reference_distr_bins = reference_distr_P;
    std::vector<int> candidate_distr_Q(&hist[0], &hist[i]);
    int num_merged_bins = i / num_quantized_bins;
    std::vector<int> candidate_distr_Q_quantized(num_quantized_bins, 0);
    int j_start = 0;
    int j_end = num_merged_bins;
    for (int idx = 0; idx < num_quantized_bins; idx++) {
      candidate_distr_Q_quantized[idx] = std::accumulate(
          &candidate_distr_Q[j_start], &candidate_distr_Q[j_end], 0);
      j_start += num_merged_bins;
      j_end += num_merged_bins;
      if ((idx + 1) == num_quantized_bins - 1) {
        j_end = i;
      }
    }
    candidate_distr_Q =
        ExpandQuantizedBins(candidate_distr_Q_quantized, reference_distr_bins);
    int Q_sum =
        std::accumulate(candidate_distr_Q.begin(), candidate_distr_Q.end(), 0);
    auto kl_divergence =
        SafeEntropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum);
    if (!kl_inited) {
      min_kl_divergence = kl_divergence;
      min_kl_index = i;
      kl_inited = true;
    } else if (kl_divergence < min_kl_divergence) {
      min_kl_divergence = kl_divergence;
      min_kl_index = i;
    } else {
    }
  }
  if (min_kl_index == 0) {
    while (starting_iter > 0) {
      if (hist[starting_iter] == 0) {
        starting_iter -= 1;
        continue;
      } else {
        break;
      }
      min_kl_index = starting_iter;
    }
  }

  LoDTensor scale_tensor;
  scale_tensor.Resize({1});
  auto* scale_ptr = scale_tensor.mutable_data<float>(CPUPlace());

  scale_ptr[0] = static_cast<float>((min_kl_index + 0.5f) * bin_width);

  return std::make_pair(quant_max, scale_tensor);
}

}  // namespace

// Run single warmup iteration
bool Quantizer::RunWarmup() {
  VLOG(3) << "Predictor: run a quantization warmup iteration";
  auto warmup_data = config_->warmup_data();

  PADDLE_ENFORCE_NOT_NULL(warmup_data,
                          "Warmup data cannot be NULL in the config.");

  std::vector<PaddleTensor> output_slots;

  // Run the inference program
  predictor_run_(*warmup_data.get(), &output_slots,
                 config_->warmup_batch_size());

  return true;
}

// Gather data from variables and calculate scales for them
bool Quantizer::CalculateScales() {
  /*
   *   std::map<std::string, std::map<std::string, LoDTensor>> gathered_data;
   *   for (auto *op : infer_program_->Block(0).AllOps()) {
   *     if (op->HasAttr("use_quantizer") && op->Attr<bool>("use_quantizer")) {
   *       const VariableNameMap &connections = op->Inputs();
   *       const VariableNameMap &connections_out = op->Outputs();
   *       connections.insert(connections.end(), connections_out.begin(),
   *                          connections_out.end());
   *
   *       for (auto &conn_name : connections) {
   *         Variable *var = scope_.FindVar(var_name);
   *         PADDLE_ENFORCE(var, "%s is not in the scope", var_name);
   *         PADDLE_ENFORCE(var->IsType<LoDTensor>(),
   *                        "Only support lod tensor now.");
   *         LoDTensor *var_tensor = var->GetMutable<LoDTensor>();
   *
   *         CalculateSingleScale(...);
   *
   *         //...
   *       }
   *     }
   *   }
   */
  return true;
}

void Quantizer::CalculateSingleScale(const std::string& op_name,
                                     const std::string& conn_name,
                                     const std::string& var_name,
                                     const LoDTensor& var_tensor) {
  // adds pairs variable name -> LoDTensor with scale to the scales map

  if (var_tensor.numel() == 0)
    throw std::runtime_error(
        "Quantizer: LoDTensor of variable for quantization should not be "
        "empty.");

  auto rule = config_->scale_algo(op_name, conn_name);
  switch (rule) {
    case ScaleAlgo::NONE:
      return;
    case ScaleAlgo::MAX: {
      scales_[var_name] = GetMaxScalingFactor(var_tensor);
      break;
    }
    case ScaleAlgo::KL:
      scales_[var_name] = GetKLScalingFactor(var_tensor);
      break;
    default:
      throw std::runtime_error("Quantizer: Unexpected ScaleAlgo specified.");
  }
}

bool Quantizer::RunQuantizePasses() {
  // push the scales to the quantize pass
  auto cpu_quantize_pass =
      framework::ir::PassRegistry::Instance().Get("cpu_quantize_pass");
  cpu_quantize_pass->Set<VarQuantMaxAndScale>("quant_var_scales", &scales_);
  // TODO(sfraczek): How to run it?

  auto cpu_quantize_squash_pass =
      framework::ir::PassRegistry::Instance().Get("cpu_quantize_squash_pass");
  auto cpu_quantize_scale_out_pass =
      framework::ir::PassRegistry::Instance().Get(
          "cpu_quantize_scale_out_pass");
  // TODO(sfraczek): How to run it?

  return true;
}

bool Quantizer::SaveModel() {
  // TODO(wojtuss): Add saving model
  return true;
}

bool Quantizer::Quantize() {
  if (!RunWarmup()) return false;
  if (!CalculateScales()) return false;
  // run quantization and optimization passes
  // save quantized model if required
  if (!RunQuantizePasses()) return false;
  if (!SaveModel()) return false;

  return true;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
