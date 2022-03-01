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

at::Tensor cnnl_bce_internal(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight, int64_t reduction) {
  std::vector<int64_t> output_size;
  auto target_ = target;
  // to avoid target is long or int dtype
  if (target.scalar_type() == at::ScalarType::Int || target.scalar_type() == at::ScalarType::Long) {
    target_ = target.to(self.scalar_type());
  }
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target);
  bool weight_flag = weight.defined();
  auto weight_impl = target_impl;
  if (weight_flag) {
    weight_impl = getMluTensorImpl(weight);
  }
  cnnlBceLossReduction_t reduction_mode = CNNL_BCE_LOSS_MEAN;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_BCE_LOSS_NONE;
      output_size = self.sizes().vec();
      break;
    case 1:
      reduction_mode = CNNL_BCE_LOSS_MEAN;
      output_size = {};
      break;
    case 2:
      reduction_mode = CNNL_BCE_LOSS_SUM;
      output_size = {};
      break;
    default:
      LOG(ERROR) << "bce_loss reduciton mode is avaliable";
      break;
  }
  auto handle = getCurrentHandle();
  auto memory_format = self.suggest_memory_format();
  at::Tensor output = at::empty(output_size, self.options(),
                                infer_memory_format(output_size.size(), memory_format));
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor output_desc;
  self_desc.set(self);
  target_desc.set(target);
  if (weight_flag) {
    weight_desc.set(weight);
  }
  output_desc.set(output);
  auto self_ptr = self_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_ptr = target_ptr;
  if (weight_flag) {
    weight_ptr = weight_impl->cnnlMalloc();
  }
  auto output_ptr = output_impl->cnnlMalloc();
  size_t sz = 0;
  TORCH_CNNL_CHECK(cnnlGetBceLossWorkspaceSize(handle, self_desc.desc(),
                   weight_flag ? weight_desc.desc() : nullptr, &sz));
  at::Tensor workspace = at::zeros(sz,
      at::TensorOptions(at::ScalarType::Byte).device(at::Device(at::Device::Type::MLU)));
  auto ws_impl = getMluTensorImpl(workspace);
  auto ws_ptr = ws_impl->cnnlMalloc();
  TORCH_CNNL_CHECK(cnnlBceLoss(
      handle, self_desc.desc(), self_ptr,
      target_desc.desc(), target_ptr, weight_flag ? weight_desc.desc() : nullptr,
      weight_flag ? weight_ptr : nullptr,
      reduction_mode, ws_ptr,
      sz, output_desc.desc(), output_ptr));
  return output;
}

at::Tensor cnnl_bce_bp_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction) {
  auto target_ = target;
  // to avoid target is long or int dtype
  if (target.scalar_type() == at::ScalarType::Int || target.scalar_type() == at::ScalarType::Long) {
    target_ = target.to(self.scalar_type());
  }
  at::Tensor grad = grad_output;
  bool weight_flag = weight.defined();
  auto grad_output_impl = getMluTensorImpl(grad);
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target_);
  decltype(target_impl) weight_impl = nullptr;
  if (weight_flag) {
    weight_impl = getMluTensorImpl(weight);
  }
  cnnlBceLossReduction_t reduction_mode = CNNL_BCE_LOSS_MEAN;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_BCE_LOSS_NONE;
      break;
    case 1:
      reduction_mode = CNNL_BCE_LOSS_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_BCE_LOSS_SUM;
      break;
    default:
      LOG(ERROR) << "bce_bp reduciton mode is unavaliable";
      break;
  }
  auto handle = getCurrentHandle();
  at::Tensor grad_input = at::empty_like(self);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  CnnlTensorDescriptor grad_output_desc;
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor grad_input_desc;

  grad_output_desc.set(grad_output);
  self_desc.set(self);
  target_desc.set(target_);
  if (weight_flag) {
    weight_desc.set(weight);
  }
  grad_input_desc.set(grad_input);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto self_ptr = self_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  decltype(target_ptr) weight_ptr = nullptr;
  if (weight_flag) {
    weight_ptr = weight_impl->cnnlMalloc();
  }
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  size_t sz = 0;
  TORCH_CNNL_CHECK(cnnlGetBceLossBackwardWorkspaceSize(handle, target_desc.desc(),
                   weight_flag ? weight_desc.desc() : nullptr, &sz));
  at::Tensor workspace = at::zeros(sz,
      at::TensorOptions(at::ScalarType::Float).device(at::Device(at::Device::Type::MLU)));
  auto ws_impl = getMluTensorImpl(workspace);
  auto ws_ptr = ws_impl->cnnlMalloc();
  TORCH_CNNL_CHECK(cnnlBceLossBackward(
      handle, grad_output_desc.desc(), grad_output_ptr,
      self_desc.desc(), self_ptr,
      target_desc.desc(), target_ptr, weight_flag ? weight_desc.desc() : nullptr,
      weight_flag ? weight_ptr : nullptr,
      reduction_mode, ws_ptr, sz, grad_input_desc.desc(),
      grad_input_ptr));
  return grad_input;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

