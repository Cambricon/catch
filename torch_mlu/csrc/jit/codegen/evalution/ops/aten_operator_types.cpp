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

static auto registry = Registerer()
    .op(c10::Symbol::fromQualString("aten::warn"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params) -> c10::optional<torch::jit::IValue> {
          return {};
        })
    .op(c10::Symbol::fromQualString("aten::__is__"),
        [](codegen::MagicmindHandle*, const torch::jit::Node* node, values_map& params) -> c10::optional<torch::jit::IValue> {
          auto input_1 = params.at(node->input(0));
          auto input_2 = params.at(node->input(1));

          return input_1.isSameIdentity(input_2);
        },
        {
          "aten::__is__(t1 self, t2 obj) -> bool",
        });

}  // namespace evalution
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
