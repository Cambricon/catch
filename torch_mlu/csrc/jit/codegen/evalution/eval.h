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

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace evalution {

using values_map = std::unordered_map<const torch::jit::Value*, torch::jit::IValue>;
using NodeEvalFunction = std::function<c10::optional<torch::jit::IValue>(
                                       codegen::MagicmindHandle*, const torch::jit::Node*, values_map&)>;
using NodeEval = std::pair<NodeEvalFunction, std::set<std::string>>;

class EvalFunctionMap final {
 public:
  ~EvalFunctionMap() {}

  static EvalFunctionMap& singleton();

  void register_node_eval(torch::jit::NodeKind kind, NodeEval func);

  bool find_node_eval(torch::jit::NodeKind kind);

  NodeEval get_node_eval(torch::jit::NodeKind kind);

 private:
  EvalFunctionMap() {}

  std::unordered_map<torch::jit::NodeKind, NodeEval> eval_function_map;
};

class Registerer final {
 public:
  Registerer() = default;
  Registerer(const Registerer&) = delete;
  Registerer& operator=(const Registerer&) = delete;
  Registerer(Registerer&&) = default;
  Registerer& operator=(Registerer&&) = default;

  Registerer&& op(torch::jit::NodeKind kind,
                  NodeEvalFunction func,
                  std::set<std::string> valid_schemas = {}) && {
    static auto& handle = EvalFunctionMap::singleton();
    handle.register_node_eval(
        kind,
        std::make_pair<NodeEvalFunction&, std::set<std::string>&>(func, valid_schemas));

    return std::move(*this);
  }
};

NodeEvalFunction findOrGetNodeEval(const torch::jit::Node* node);

bool isEvalNode(const torch::jit::Node* node);

c10::optional<torch::jit::IValue> evalNode(codegen::MagicmindHandle *handle,
                                           const torch::jit::Node* node);

}  // namespace evalution
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
