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

void cnnl_adaptive_avg_pool_backward_internal(
  at::Tensor& grad_input, const at::Tensor& grad_output, const at::Tensor& input) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  if (input.dim() == 5) layout = CNNL_LAYOUT_NDHWC;
  input_desc.set(grad_output, layout);
  output_desc.set(grad_input, layout);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(grad_output);
  auto output_impl = getMluTensorImpl(grad_input);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingBackward(handle, input_desc.desc(), input_ptr,
    nullptr, nullptr, CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, output_desc.desc(), output_ptr));
}

void cnnl_adaptive_max_pool2d_backward_internal(
  at::Tensor& grad_input, const at::Tensor& grad_output,
  const at::Tensor& input, const at::Tensor& indices) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor indices_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(grad_output, CNNL_LAYOUT_NHWC);
  indices_desc.set(indices, CNNL_LAYOUT_NHWC);
  output_desc.set(grad_input, CNNL_LAYOUT_NHWC);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(grad_output);
  auto indices_impl = getMluTensorImpl(indices);
  auto output_impl = getMluTensorImpl(grad_input);
  auto input_ptr = input_impl->cnnlMalloc();
  auto indices_ptr = indices_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingBackward(handle, input_desc.desc(), input_ptr,
    indices_desc.desc(), indices_ptr, CNNL_POOLING_MAX, output_desc.desc(), output_ptr));
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
