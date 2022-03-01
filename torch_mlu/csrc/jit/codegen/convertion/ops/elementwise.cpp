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

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace convertion {

inline bool process_elementwise_op(codegen::MagicmindHandle* handle,
                                   const torch::jit::Node* node,
                                   torch::jit::Stack& params,
                                   const magicmind::IElementwise node_type) {
  auto self = codegen::getOrCreateITensor(handle, params[0]);
  auto other = codegen::getOrCreateITensor(handle, params[1]);

  auto input_dtype1 = self->GetDataType();
  auto input_dtype2 = other->GetDataType();
  auto common_itensor_type = input_dtype1;

  if (input_dtype1 != input_dtype2) {
    if (params[0].isScalar()) {
      // For add.Scalar, the dtype of tensor from scalar value should be aligned to input tensor.
      auto common_type = at::native::result_type(
          codegen::getEmptyTensor(other),
          params[0].toScalar());
      common_itensor_type = utils::scalarTypeToMagicmindDataType(common_type);
      auto cast = handle->network->AddICastNode(
          self, common_itensor_type);
      self = cast->GetOutput(0);
    } else if (params[1].isScalar()) {
      // For add.Scalar, the dtype of tensor from scalar value should be aligned to input tensor.
      auto common_type = at::native::result_type(
          codegen::getEmptyTensor(self),
          params[1].toScalar());
      common_itensor_type = utils::scalarTypeToMagicmindDataType(common_type);
      auto cast = handle->network->AddICastNode(
          other, common_itensor_type);
      other = cast->GetOutput(0);
    } else {
      auto common_type = at::native::result_type(
          codegen::getEmptyTensor(self),
          codegen::getEmptyTensor(other));
      common_itensor_type = utils::scalarTypeToMagicmindDataType(common_type);
      if (common_itensor_type == input_dtype1) {
        auto cast = handle->network->AddICastNode(
            other, utils::scalarTypeToMagicmindDataType(common_type));
        other = cast->GetOutput(0);
      } else {
        auto cast = handle->network->AddICastNode(
            self, utils::scalarTypeToMagicmindDataType(common_type));
        self = cast->GetOutput(0);
      }
    }
  }

  auto elementwise_op = handle->network->AddIElementwiseNode(self, other, node_type);
  MM_CHECK(elementwise_op->SetOutputType(0, common_itensor_type));
  if (params.size() > 2) {
    auto alpha = params[2].toScalar().to<float>();
    if (alpha != 1.0) {
      MM_CHECK(elementwise_op->SetAlpha1(1.0));
      MM_CHECK(elementwise_op->SetAlpha2(alpha));
    }
  }
  auto output_tensor = elementwise_op->GetOutput(0);
  handle->bindingValueAndIvalue(
          node->outputs()[0], codegen::bindITensor(output_tensor));
  return true;
}

static auto registry = Registerer()
    .op("aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::ADD);
        })
    .op("aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::ADD);
        })
    .op("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::MUL);
        })
    .op("aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::MUL);
        })
    .op("aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::SUB);
        })
    .op("aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          return process_elementwise_op(handle, node, params, magicmind::IElementwise::SUB);
        })
    .op("aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          auto self = codegen::getOrCreateITensor(handle, params[0]);
          auto other = codegen::getOrCreateITensor(handle, params[1]);
          auto div_op = handle->network->AddIDivNode(self, other);
          auto output_dtype = self->GetDataType();
          MM_CHECK(div_op->SetOutputType(0, output_dtype));
          auto output = div_op->GetOutput(0);
          handle->bindingValueAndIvalue(
                  node->outputs()[0], codegen::bindITensor(output));
          return true;
        });

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
