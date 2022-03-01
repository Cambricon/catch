/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "jit/utils/utils.h"
#include "aten/util/cnlog.h"

namespace torch_mlu {
namespace jit {
namespace utils {

std::string getOpFullNameFromNode(const torch::jit::Node* node) {
  std::string name = node->kind().toQualString();
  auto *maybe_schema = node->maybeSchema();
  std::string overload_name = "";
  if (maybe_schema) {
     overload_name = node->schema().overload_name();
  }

  if (overload_name == "") return name;
  return name + "." + overload_name;
}

void saveTensor(const at::Tensor ival, const std::string& file) {
  auto content = torch::jit::pickle_save(ival);
  std::ofstream out(file);
  std::ostreambuf_iterator<char> osb(out);
  auto in = content.begin();
  while (in != content.end()) {
    *osb++ = *in++;
  }
}

at::Tensor loadTensor(const std::string& file) {
  std::vector<char> input;
  std::ifstream in;
  in.open(file);
  std::istreambuf_iterator<char> isb(in);
  input.insert(
        input.begin(),
        isb,
        std::istreambuf_iterator<char>());
  auto ival = torch::jit::pickle_load(input);
  return ival.toTensor();
}

void* accessTensor(const at::Tensor &tensor) {
  if (!tensor.defined()) {
    CNLOG(DBG) << "Logic Error! Undefined tensor must be handled by caller";
    return nullptr;
  }
  using c10::ScalarType;
  return tensor.data_ptr();
}

void setQuantizedParams(float scale, int qmode, int use_symmetry, std::vector<float>& params) {
  // TODO: Support asymmetry quantization later.
  if (use_symmetry != 1) {
    CNLOG(ERROR) << "Asymmetric quantization is not supported currently.";
    return;
  }
  else {
    float scale_ = 1, pos_ = 1, offset_ = 0, scale_factor_ = 1.0;
    int n = 8;
    int qmin = -128, qmax = 127;
    if (qmode == 2 || qmode == 4) {
      n = 16;
      qmin = -32768;
      qmax =  32767;
    }

    float absmax = qmax / scale;
    pos_ = std::floor(std::log(absmax) / std::log(2)) - (n - 2);
    scale_ = std::pow(2, pos_) * (std::pow(2, (n -1)) - 1) / absmax;
    scale_factor_ = std::pow(2, pos_) / scale_;
    params.emplace_back(scale_factor_);
    params.emplace_back(offset_);
    return;
  }
}

void setQuantizedParamsPerAxis(const std::vector<float>& scales,
                               int qmode,
                               int use_symmetry,
                               std::vector<float>& params) {
  // First 2 elements are input scale and invalid weight scale.
  for (int i = 2; i < scales.size(); i++) {
    setQuantizedParams(scales[i], qmode, use_symmetry, params);
  }
  return;
}

void getDynamicRange(const std::vector<float>& scales,
                     int qmode,
                     int use_symmetry,
                     size_t bitwidth,
                     std::vector<magicmind::Range>& input_ranges,
                     std::vector<magicmind::Range>& weight_ranges) {
  input_ranges.clear();
  weight_ranges.clear();
  bool b_per_axis = (qmode > 2 && qmode < 5) ? true : false;

  // calculate scale_factor and zero_point for input and weight
  std::vector<float> input_params;
  std::vector<float> weight_params;
  setQuantizedParams(scales[0], qmode, use_symmetry, input_params);
  if (b_per_axis) {
    setQuantizedParamsPerAxis(scales, qmode, use_symmetry, weight_params);
  } else {
    setQuantizedParams(scales[1], qmode, use_symmetry, weight_params);
  }

  // set dynamic ranges for input quantization
  auto tmp_range1 =
    magicmind::UniformQuantParamToRangeWithQuantAlg({input_params[0], (long)input_params[1]},
                                                    bitwidth,
                                                    "symmetric");
  input_ranges.emplace_back(tmp_range1);

  // set dynamic ranges for weight quantization
  if (!b_per_axis) {
    // quantization by tensor
    auto tmp_range2 =
      magicmind::UniformQuantParamToRangeWithQuantAlg({weight_params[0], (long)weight_params[1]},
                                                      bitwidth,
                                                      "symmetric");
    weight_ranges.emplace_back(tmp_range2);
  } else {
    // quantization by channel
    for (int i = 0; i < weight_params.size() /2; i++) {
      auto tmp_range3 =
        magicmind::UniformQuantParamToRangeWithQuantAlg(
                   {weight_params[2 * i], (long)weight_params[2 * i + 1]},
                   bitwidth,
                   "symmetric");
      weight_ranges.emplace_back(tmp_range3);
    }
  }
  input_params.clear();
  weight_params.clear();
  return;
}

} // namespace utils
} // namespace jit
} // namespace torch_mlu
