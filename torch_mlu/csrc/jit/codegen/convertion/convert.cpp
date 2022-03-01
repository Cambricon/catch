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

#include "jit/codegen/convertion/convert.h"
#include "jit/codegen/evalution/eval.h"

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace convertion {

ConvertFunctionMap& ConvertFunctionMap::singleton() {
  static ConvertFunctionMap _singleton;
  return _singleton;
}

void ConvertFunctionMap::register_node_convert(const std::string signature,
                                               NodeConvertFunction func) {
  auto schema = torch::jit::parseSchemaOrName(signature);
  auto operator_name = (schema.is_left()) ? schema.left() : schema.right().operator_name();

  if (convert_function_map.find(operator_name) != convert_function_map.end()) {
    TORCH_WARN(signature, " node kind has already been registered into ConvertFunctionMAP,",
               " it will be overrided now.");
  }

  convert_function_map[operator_name] = std::move(func);
}

bool ConvertFunctionMap::find_node_convert(const torch::jit::FunctionSchema* schema) {
  if (!schema) return false;

  auto operator_name = schema->operator_name();

  if (convert_function_map.find(operator_name) != convert_function_map.end()) {
    return true;
  } else {
    return false;
  }
}

NodeConvertFunction ConvertFunctionMap::get_node_convert(
    const torch::jit::FunctionSchema* schema) {
  if (!schema) return nullptr;

  auto operator_name = schema->operator_name();

  if (find_node_convert(schema)) {
    return convert_function_map[operator_name];
  } else {
    TORCH_WARN(schema->name(), " node kind has not been registered into ConvertFunctionMAP.");
    return nullptr;
  }
}

NodeConvertFunction findOrGetNodeConvert(const torch::jit::Node* node) {
  if (!node) return nullptr;

  auto handle = ConvertFunctionMap::singleton();
  auto schema = node->maybeSchema();

  if (!schema) return nullptr;

  if (!handle.find_node_convert(schema)) {
    return nullptr;
  }

  return handle.get_node_convert(schema);
}

bool isConvertNode(const torch::jit::Node* node) {
  if (findOrGetNodeConvert(node)) {
    return true;
  } else {
    return false;
  }
}

void convertNode(codegen::MagicmindHandle* handle, const torch::jit::Node* node) {
  TORCH_CHECK(handle != nullptr && node != nullptr,
      "handle and node pointer should not be nullptr for convertNode().");

  torch::jit::Stack params;

  for (auto value : node->inputs()) {
    if (handle->conversion_value_map.find(value) != handle->conversion_value_map.end()) {
      params.push_back(handle->conversion_value_map[value]);
    } else if (evalution::isEvalNode(value->node())) {
      auto out = evalution::evalNode(handle, value->node());
      if (out) {
        handle->bindingValueAndIvalue(value, out.value());
      }
      params.push_back(out.value());
    } else {
      AT_ERROR(node->maybeSchema()->name(),
               " node kind can't be converted to magicmind node!!!");
    }
  }

  TORCH_CHECK(params.size() == node->inputs().size(),
              "the node input number can't match.");

  auto convert_func = findOrGetNodeConvert(node);
  auto result = convert_func(handle, node, params);
  TORCH_CHECK(result, "Convert node error.")
}

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
