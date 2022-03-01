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

#include "torch/csrc/jit/ir/constants.h"

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace evalution {

static auto registry = Registerer()
    .op(c10::Symbol::fromQualString("prim::Constant"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          return torch::jit::toIValue(node->output());
        })
    .op(c10::Symbol::fromQualString("prim::NumToTensor"),
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          TORCH_CHECK(handle->conversion_value_map.count(node->input(0)) != 0,
                      "Input of NumToTensor is not in the symbol map");
          return handle->conversion_value_map.at(node->input(0));
        })
    .op(c10::Symbol::fromQualString("prim::ListConstruct"),
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          const auto num_inputs = node->inputs().size();
          c10::ListTypePtr lt = node->output()->type()->expect<c10::ListType>();
          std::vector<magicmind::ITensor*> list;
          for (auto in : node->inputs()) {
            auto ivalue = handle->conversion_value_map[in];
            params[in] = bindITensor(getOrCreateITensor(handle, ivalue));
            handle->bindingValueAndIvalue(in, params[in]);
            TORCH_CHECK(isITensor(params.at(in)),
                        "Input of ListConstruct is not in the symbol map");
            list.emplace_back(getITensor(params.at(in)));
          }
          if (lt->getElementType()->isSubtypeOf(torch::jit::TensorType::get())) {
            return c10::optional<torch::jit::IValue>(codegen::bindITensorVector(list));
          } else {
            int dim_val = 0;
            at::Scalar dim{dim_val};
            auto dim_itensor = createConstITensor(handle, at::scalar_to_tensor(dim));
            auto pack_op = handle->network->AddIPackNode(dim_itensor, list);
            auto output_dtype = list[0]->GetDataType();
            MM_CHECK(pack_op->SetOutputType(0, output_dtype));
            auto output_tensor = pack_op->GetOutput(0);
            return c10::optional<torch::jit::IValue>(codegen::bindITensor(output_tensor));
          }
        })
    .op(c10::Symbol::fromQualString("prim::GetAttr"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          const auto type = node->input()->type()->expect<c10::ClassType>();
          const auto& field = node->s(torch::jit::attr::name);
          const auto slot = type->getAttributeSlot(field);

          auto ivalue = params.at(node->input(0));
          auto userObj = ivalue.toObject();
          auto value = userObj->getSlot(slot);

          return std::move(value);
        })
    .op(c10::Symbol::fromQualString("prim::Uninitialized"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          return c10::IValue::uninitialized();
        })
    .op(c10::Symbol::fromQualString("prim::shape"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params)
            -> c10::optional<torch::jit::IValue> {
          auto ivalue = params.at(node->input(0));
          if (codegen::isITensor(ivalue)) {
            auto itensor = codegen::getITensor(ivalue);
            auto dims = itensor->GetDimension().GetDims();
            return dims;
          } else {
            auto tensor = ivalue.to<at::Tensor>();
            return tensor.sizes();
          }
        },
        {
          "prim::shape(Tensor a) -> (int[])",
        });

}  // namespace evalution
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
