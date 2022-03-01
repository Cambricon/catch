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

at::Tensor& cnnl_upsample_internal(at::Tensor& output, const at::Tensor& self,
                                   at::IntArrayRef output_size, bool align_corners,
                                   bool align_center, cnnlInterpMode_t interp_mode) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(self, CNNL_LAYOUT_NHWC);
  descOutput.set(output, CNNL_LAYOUT_NHWC);

  // malloc mlu memory
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlInterp(handle, align_corners, align_center,
                           interp_mode, descInput.desc(), input_ptr,
                           descOutput.desc(), output_ptr));
  return output;
}

at::Tensor cnnl_upsample_backward_internal(at::Tensor& grad_input, const at::Tensor& grad_output,
                                           at::IntArrayRef output_size, at::IntArrayRef input_size,
                                           bool align_corners, bool align_center,
                                           cnnlInterpBackwardMode_t interp_mode) {
  CnnlTensorDescriptor descGradInput;
  CnnlTensorDescriptor descGradOutput;
  descGradInput.set(grad_input, CNNL_LAYOUT_NHWC);
  descGradOutput.set(grad_output, CNNL_LAYOUT_NHWC);

  // malloc mlu memory
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlInterpBackward(handle, align_corners, align_center, interp_mode,
                                      descGradOutput.desc(), grad_output_ptr,
                                      descGradInput.desc(), grad_input_ptr));
  return grad_input;
}

}   // namespace ops
}   // namespace cnnl
}   // namespace torch_mlu
