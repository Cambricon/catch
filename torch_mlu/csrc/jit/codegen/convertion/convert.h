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

#pragma once

#include <functional>
#include <unordered_map>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#include <ATen/core/ivalue.h>

#include "jit/codegen/codegen.h"
#include "jit/codegen/tensor.h"
#include "jit/utils/utils.h"

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace convertion {

using NodeConvertFunction = std::function<bool(codegen::MagicmindHandle*,
                                               const torch::jit::Node*,
                                               torch::jit::Stack&)>;

class ConvertFunctionMap final {
 public:
  ~ConvertFunctionMap() {}

  static ConvertFunctionMap& singleton();

  void register_node_convert(const std::string signature, NodeConvertFunction func);

  bool find_node_convert(const torch::jit::FunctionSchema* schema);

  NodeConvertFunction get_node_convert(const torch::jit::FunctionSchema* schema);

 private:
  ConvertFunctionMap() {}

  std::unordered_map<c10::OperatorName, NodeConvertFunction> convert_function_map;
};

class Registerer final {
 public:
  Registerer() = default;
  Registerer(const Registerer&) = delete;
  Registerer& operator=(const Registerer&) = delete;
  Registerer(Registerer&&) = default;
  Registerer& operator=(Registerer&&) = default;

  Registerer&& op(const std::string signature,
                  NodeConvertFunction func) && {
    static auto& handle = ConvertFunctionMap::singleton();
    handle.register_node_convert(signature, func);

    return std::move(*this);
  }
};

NodeConvertFunction findOrGetNodeConvert(const torch::jit::Node* node);

bool isConvertNode(const torch::jit::Node* node);

void convertNode(codegen::MagicmindHandle*, const torch::jit::Node* node);

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
