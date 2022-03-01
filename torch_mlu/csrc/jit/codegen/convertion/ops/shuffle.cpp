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

static auto registry = Registerer()
    .op("aten::reshape(Tensor self, int[] shape) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto self = codegen::getOrCreateITensor(handle, params[0]);
          auto shape = codegen::getOrCreateITensor(handle, params[1]);

          magicmind::ITensor* scale_factors = nullptr;
          auto reshape_op = handle->network->AddIReshapeNode(self, shape);

          MM_CHECK(reshape_op->SetAxis(0));
          MM_CHECK(reshape_op->SetNumAxes(-1));
          MM_CHECK(reshape_op->SetAllowZero(true));

          auto output_dtype = self->GetDataType();
          MM_CHECK(reshape_op->SetOutputType(0, output_dtype));
          auto output_tensor = reshape_op->GetOutput(0);

          handle->bindingValueAndIvalue(
                node->outputs()[0], codegen::bindITensor(output_tensor));
            return true;
        })
    .op("aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto output_dtype = input_tensor->GetDataType();
          auto dim0 = codegen::getOrCreateITensor(handle, params[1]);
          auto dim1 = codegen::getOrCreateITensor(handle, params[2]);

          auto t = handle->network->AddITransposeNode(input_tensor, dim0, dim1);
          MM_CHECK(t->SetOutputType(0, output_dtype));
          auto output_tensor = t->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto output_dtype = input_tensor->GetDataType();
          auto dim0 = codegen::getOrCreateITensor(handle, params[1]);
          auto dim1 = codegen::getOrCreateITensor(handle, params[2]);

          auto t = handle->network->AddITransposeNode(input_tensor, dim0, dim1);
          MM_CHECK(t->SetOutputType(0, output_dtype));
          auto output_tensor = t->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::t(Tensor(a) self) -> Tensor(a)",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto output_dtype = input_tensor->GetDataType();
          // MagicMind maps operator t(input) to transpose(input,0,1).
          torch::jit::IValue dim0_tmp(0);
          torch::jit::IValue dim1_tmp(1);
          // Input dim is 1:: TODO
          // if (input_tensor->GetDimension().GetDimsNum() == 1) {
          //     dim1_tmp = torch::jit::IValue(0);
          // }
          auto dim0 = dim0_tmp.isScalar() ?
                codegen::getOrCreateITensor(handle, dim0_tmp) : nullptr;
          auto dim1 = dim0_tmp.isScalar() ?
                codegen::getOrCreateITensor(handle, dim1_tmp) : nullptr;

          auto t = handle->network->AddITransposeNode(input_tensor, dim0, dim1);
          MM_CHECK(t->SetOutputType(0, output_dtype));
          auto output_tensor = t->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::t_(Tensor(a!) self) -> Tensor(a!)",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto output_dtype = input_tensor->GetDataType();
          // MagicMind maps operator t(input) to transpose(input,0,1).
          torch::jit::IValue dim0_tmp(0);
          torch::jit::IValue dim1_tmp(1);
          // Input dim is 1:: TODO
          // if (input_tensor->GetDimension().GetDimsNum() == 1) {
          //     dim1_tmp = torch::jit::IValue(0);
          // }
          auto dim0 = dim0_tmp.isScalar() ?
                codegen::getOrCreateITensor(handle, dim0_tmp) : nullptr;
          auto dim1 = dim0_tmp.isScalar() ?
                codegen::getOrCreateITensor(handle, dim1_tmp) : nullptr;

          auto t = handle->network->AddITransposeNode(input_tensor, dim0, dim1);
          MM_CHECK(t->SetOutputType(0, output_dtype));
          auto output_tensor = t->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto self = codegen::getOrCreateITensor(handle, params[0]);
          auto start_dim = codegen::getOrCreateITensor(handle, params[1]);
          auto end_dim = codegen::getOrCreateITensor(handle, params[2]);
          auto output_dtype = self->GetDataType();

          auto flatten_op = handle->network->AddIFlattenNode(
              self, start_dim, end_dim);
          MM_CHECK(flatten_op->SetOutputType(0, output_dtype));
        auto output_tensor = flatten_op->GetOutput(0);
        handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        });

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
