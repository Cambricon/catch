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

#include "jit/codegen/evalution/eval.h"
#include "jit/codegen/tensor.h"

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace evalution {

EvalFunctionMap& EvalFunctionMap::singleton() {
  static EvalFunctionMap _singleton;
  return _singleton;
}

void EvalFunctionMap::register_node_eval(torch::jit::NodeKind kind,
                                         NodeEval func) {
  if (eval_function_map.find(kind) != eval_function_map.end()) {
    TORCH_WARN(kind.toQualString(), " node kind has already been registered into EvalFunctionMAP,",
               " it will be overrided now.");
  }

  eval_function_map[kind] = std::move(func);
}

bool EvalFunctionMap::find_node_eval(torch::jit::NodeKind kind) {
  if (eval_function_map.find(kind) != eval_function_map.end()) {
    return true;
  } else {
    return false;
  }
}

NodeEval EvalFunctionMap::get_node_eval(torch::jit::NodeKind kind) {
  if (find_node_eval(kind)) {
    return eval_function_map[kind];
  } else {
    TORCH_WARN(kind.toQualString(), " node kind has not been registered into EvalFunctionMAP.");
    return std::pair<NodeEvalFunction, std::set<std::string>>();
  }
}

NodeEvalFunction findOrGetNodeEval(const torch::jit::Node* node) {
  TORCH_CHECK(node != nullptr, "node pointer should not be nullptr in findOrGetNodeEval().");

  auto handle = EvalFunctionMap::singleton();
  auto node_kind = node->kind();

  if (!handle.find_node_eval(node_kind)) {
    return nullptr;
  }

  auto node_eval = handle.get_node_eval(node_kind);
  auto node_vaild_schema = node_eval.second;

  if (node_vaild_schema.size() > 0) {
    auto schema = node->maybeSchema();

    for (auto valid_schema : node_vaild_schema) {
      if (torch::jit::parseSchema(valid_schema).operator_name() == schema->operator_name()) {
        return node_eval.first;
      }
    }

    return nullptr;
  } else {
    return node_eval.first;
  }
}

bool isEvalNode(const torch::jit::Node* node) {
  if (findOrGetNodeEval(node)) {
    return true;
  } else {
    return false;
  }
}

c10::optional<torch::jit::IValue> evalNode(codegen::MagicmindHandle *handle,
                                           const torch::jit::Node* node) {
  TORCH_CHECK(handle != nullptr && node != nullptr,
      "handle and node pointer should not be nullptr for evalNode().");

  values_map params;
  for (auto value : node->inputs()) {
    if (params.find(value) != params.end()) continue;
    if (handle->conversion_value_map.find(value) != handle->conversion_value_map.end()) {
      // auto ivalue = handle->conversion_value_map[value];
      // params[value] = bindITensor(getOrCreateITensor(handle, ivalue));
      // handle->bindingValueAndIvalue(value, params[value]);
      params[value] = handle->conversion_value_map[value];

    } else if (isEvalNode(value->node())) {
      auto out = evalNode(handle, value->node());
      if (out) {
        auto ivalue = out.value();
        params[value] = ivalue;
        handle->bindingValueAndIvalue(value, ivalue);
      }
    } else {
      AT_ERROR("Can't evaluate the node!!!");
      return {};
    }
  }

  auto eval_func = findOrGetNodeEval(node);
  return eval_func(handle, node, params);
}

}  // namespace evalution
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
