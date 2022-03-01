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

magicmind::ITensor* batch_norm_forward(
    codegen::MagicmindHandle* handle, magicmind::ITensor* input,
    magicmind::ITensor* scale, magicmind::ITensor* offset,
    magicmind::ITensor* running_mean, magicmind::ITensor* running_var,
    double eps) {
  auto batch_norm_op = handle->network->AddIFusedBatchNormNode(input,
                                                               running_mean,
                                                               running_var,
                                                               scale,
                                                               offset);

  // param of SetAxis means the position of C dim. for example, 1 means NC(D)HW. 3 means NHWC.
  MM_CHECK(batch_norm_op->SetAxis(1));
  MM_CHECK(batch_norm_op->SetEpsilon(static_cast<float>(eps)));
  // return the output
  auto output_dtype = input->GetDataType();
  MM_CHECK(batch_norm_op->SetOutputType(0, output_dtype));
  return batch_norm_op->GetOutput(0);
  }


static auto registry = Registerer()
    .op(R"SIG(aten::batch_norm(Tensor input, Tensor? gamma, Tensor? beta,
                               Tensor? mean, Tensor? var,
                               bool training, float momentum, float eps,
                               bool cudnn_enabled) -> (Tensor))SIG",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto scale_tensor = params[1].isTensor() ? \
                     codegen::getOrCreateITensor(handle, params[1]) : nullptr;
          auto offset_tensor = params[2].isTensor() ? \
                      codegen::getOrCreateITensor(handle, params[2]) : nullptr;
          auto running_mean_tensor = params[3].isTensor() ? \
                      codegen::getOrCreateITensor(handle, params[3]) : nullptr;
          auto running_var_tensor = params[4].isTensor() ? \
                      codegen::getOrCreateITensor(handle, params[4]) : nullptr;

          // Momentum is used to calculate running mean and variance in training session,
          // so this param will not be used in here.
          auto momentum = params[6].toDouble();
          auto eps = params[7].toDouble();

          magicmind::ITensor* output_tensor = batch_norm_forward(handle,
                                                                 input_tensor,
                                                                 scale_tensor,
                                                                 offset_tensor,
                                                                 running_mean_tensor,
                                                                 running_var_tensor,
                                                                 eps);

          handle->bindingValueAndIvalue(node->outputs()[0], codegen::bindITensor(output_tensor));

          return true;
        });

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
