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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_mse_loss_internal(const at::Tensor& input,
     const at::Tensor & target, int64_t reduction) {
  std::vector<int64_t> output_size;
  cnnlMSELossReduction_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_MSE_LOSS_NONE;
      output_size = input.sizes().vec();
      break;
    case 1:
      reduction_mode = CNNL_MSE_LOSS_MEAN;
      output_size = {};
      break;
    case 2:
      reduction_mode = CNNL_MSE_LOSS_SUM;
      output_size = {};
      break;
    default:
      LOG(ERROR) << "nll_loss reduciton mode is avaliable";
      break;
  }
  at::Tensor output = at::empty(output_size, input.options(), c10::MemoryFormat::Contiguous);
  
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descTarget;
  descInput.set(input);
  descOutput.set(output);
  descTarget.set(target);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto target_impl = getMluTensorImpl(target);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();
  


  TORCH_CNNL_CHECK(cnnlMSELoss(handle,
                               reduction_mode,
                               descInput.desc(),
                               input_ptr,
                               descTarget.desc(),
                               target_ptr,
                               descOutput.desc(),
                               output_ptr));
  return output;
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
