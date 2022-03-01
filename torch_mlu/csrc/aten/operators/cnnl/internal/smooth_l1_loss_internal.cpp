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

at::Tensor& cnnl_smooth_l1_loss_forward_internal(at::Tensor& output,
                                                 const at::Tensor& self,
                                                 const at::Tensor& target,
                                                 int64_t reduction) {
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target);
  auto output_impl = getMluTensorImpl(output);

  // prepare reduction_mode
  cnnlSmoothL1LossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
      break;
    default:
      LOG(ERROR) << "smooth_l1_loss reduciton mode is avaliable";
      break;
  }

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor output_desc;
  self_desc.set(self);
  target_desc.set(target);
  output_desc.set(output);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto self_ptr = self_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // calculate
  TORCH_CNNL_CHECK(cnnlSmoothL1LossForward(handle,
                                           self_desc.desc(),
                                           self_ptr,
                                           target_desc.desc(),
                                           target_ptr,
                                           output_desc.desc(),
                                           output_ptr,
                                           reduction_mode));
  return output;
}

at::Tensor& cnnl_smooth_l1_loss_backward_internal(at::Tensor& grad_input,
                                                  const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  const at::Tensor& target,
                                                  int64_t reduction) {
  auto self_impl = getMluTensorImpl(self);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto target_impl = getMluTensorImpl(target);
  auto grad_input_impl = getMluTensorImpl(grad_input);

  cnnlSmoothL1LossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
      break;
    default:
      LOG(ERROR) << "smooth_l1_loss reduciton mode is avaliable";
      break;
  }

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor grad_output_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor grad_input_desc;
  self_desc.set(self);
  grad_output_desc.set(grad_output);
  target_desc.set(target);
  grad_input_desc.set(grad_input);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto self_ptr = self_impl->cnnlMalloc();
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();

  // calculate
  TORCH_CNNL_CHECK(cnnlSmoothL1LossBackward(handle,
                                            self_desc.desc(),
                                            self_ptr,
                                            target_desc.desc(),
                                            target_ptr,
                                            grad_output_desc.desc(),
                                            grad_output_ptr,
                                            grad_input_desc.desc(),
                                            grad_input_ptr,
                                            reduction_mode));

  return grad_input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
