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

#include "jit/codegen/io.h"
#include "jit/codegen/tensor.h"
#include "jit/utils/data_type.h"

namespace torch_mlu {
namespace jit {
namespace codegen {

void addStaticParameters(MagicmindHandle *handle,
                         at::ArrayRef<const torch::jit::Value*> inputs,
                         const at::ArrayRef<torch::jit::IValue>& stack) {
  TORCH_CHECK(handle != nullptr, "handle should not be nullptr for addStaticParameters().");
  TORCH_CHECK(inputs.size() == stack.size(),
              "size of block->inputs is not equal to stack.size()");

  for (size_t i = 0; i < inputs.size(); i++) {
    // tensor and tensorlist are considered as inputs
    if (stack[i].isTensor() || stack[i].isTensorList()) {
      continue;
    } else {
      handle->bindingValueAndIvalue(inputs[i], stack[i]);
    }
  }
}

bool addInputs(MagicmindHandle *handle,
               at::ArrayRef<const torch::jit::Value*> inputs,
               const at::ArrayRef<torch::jit::IValue>& stack) {
  TORCH_CHECK(handle != nullptr, "handle should not be nullptr for addInputs().");
  TORCH_CHECK(inputs.size() == stack.size(),
              "size of block->inputs is not equal to stack.size()");

  bool status = false;
  for (size_t i = 0; i < inputs.size(); i++) {
    if (stack[i].isTensor()) {
      auto tensor = stack[i].toTensor();
      auto data_type = utils::scalarTypeToMagicmindDataType(tensor.scalar_type());
      auto input_sizes = tensor.sizes().vec();
      auto input_tensor = handle->network->AddInput(data_type, magicmind::Dims(input_sizes));

      // Set the has_fp16_input when the data type of input tensor is fp16.
      if (tensor.scalar_type() == at::kHalf) handle->has_fp16_input = true;

      handle->bindingValueAndIvalue(inputs[i], codegen::bindITensor(input_tensor));
      status = true;
    } else if (stack[i].isModule()) {
      status = true;
    } else if (stack[i].isTensorList()) {
      auto tensor_vec = stack[i].toTensorVector();
      std::vector<magicmind::ITensor*> itensor_vec;
      for (auto & tensor : tensor_vec) {
        auto data_type = utils::scalarTypeToMagicmindDataType(tensor.scalar_type());
        auto input_sizes = tensor.sizes().vec();
        auto input_tensor = handle->network->AddInput(data_type, magicmind::Dims(input_sizes));
        itensor_vec.push_back(input_tensor);
      }
      handle->bindingValueAndIvalue(inputs[i], codegen::bindITensorVector(itensor_vec));
      status = true;
    }
  }

  return status;
}

bool markOutputs(MagicmindHandle *handle,
                 at::ArrayRef<const torch::jit::Value*> outputs) {
  TORCH_CHECK(handle != nullptr, "handle should not be nullptr for markOutputs().");
  int idx = 0;
  bool status = false;
  for (auto value : outputs) {
    if (handle->conversion_value_map.find(value) != handle->conversion_value_map.end()) {
      auto ivalue = handle->conversion_value_map[value];
      if (isITensor(ivalue)) {
        auto mm_tensor = getITensor(ivalue);
        handle->network->MarkOutput(mm_tensor);
        status = true;
      }
    }
  }

  return status;
}

}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
