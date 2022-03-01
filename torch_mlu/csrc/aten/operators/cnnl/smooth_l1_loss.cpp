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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_smooth_l1_loss_forward(const at::Tensor& self,
                                       const at::Tensor& target,
                                       int64_t reduction) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  std::vector<int64_t> output_size;
  at::Tensor output;
  if (reduction == 0) {
    output_size = self_contiguous.sizes().vec();
    output = at::empty(output_size,
                       self_contiguous.options(),
                       memory_format);
  } else {
    output_size = {};
    output = at::empty(output_size,
                       self_contiguous.options());
  }

  return cnnl_smooth_l1_loss_forward_internal(output,
                                              self_contiguous,
                                              target_contiguous,
                                              reduction);
}

at::Tensor& cnnl_smooth_l1_loss_forward_out(at::Tensor& output,
                                           const at::Tensor& self,
                                           const at::Tensor& target,
                                           int64_t reduction) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  if (reduction != 0) {
    resize_impl_mlu_(getMluTensorImpl(output), {}, {});
    return cnnl_smooth_l1_loss_forward_internal(output,
                                                self_contiguous,
                                                target_contiguous,
                                                reduction);
  }
  if (output.numel() >= self.numel()) {
    resize_impl_mlu_(getMluTensorImpl(output), self_contiguous.sizes(),
                     self_contiguous.strides());
    return cnnl_smooth_l1_loss_forward_internal(output,
                                                self_contiguous,
                                                target_contiguous,
                                                reduction);
  }
  auto output_new = at::native::empty_like(self_contiguous);
  cnnl_smooth_l1_loss_forward_internal(output_new,
                                       self_contiguous,
                                       target_contiguous,
                                       reduction);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(output_new));
  resize_impl_mlu_(getMluTensorImpl(output), output_new.sizes(), output_new.strides());
  return output;
}

at::Tensor cnnl_smooth_l1_loss_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output,
                                  infer_memory_format(grad_output.dim(), memory_format));

  auto grad_input = at::empty_like(self_contiguous);
  return cnnl_smooth_l1_loss_backward_internal(grad_input,
                                               grad_output_contiguous,
                                               self_contiguous,
                                               target_contiguous,
                                               reduction);
}

at::Tensor& cnnl_smooth_l1_loss_backward_out(at::Tensor& grad_input,
                                            const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Tensor& target,
                                            int64_t reduction) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output,
                                  infer_memory_format(grad_output.dim(), memory_format));

  if (grad_input.numel() >= self.numel()) {
    resize_impl_mlu_(getMluTensorImpl(grad_input), self_contiguous.sizes(),
                     self_contiguous.strides());
    return cnnl_smooth_l1_loss_backward_internal(grad_input,
                                                 grad_output_contiguous,
                                                 self_contiguous,
                                                 target_contiguous,
                                                 reduction);
  }
  auto grad_input_new = at::empty_like(self_contiguous);
  cnnl_smooth_l1_loss_backward_internal(grad_input_new,
                                        grad_output_contiguous,
                                        self_contiguous,
                                        target_contiguous,
                                        reduction);
  getMluTensorImpl(grad_input)->copy_cnnl_metadata_from(getMluTensorImpl(grad_input_new));
  resize_impl_mlu_(getMluTensorImpl(grad_input), grad_input_new.sizes(), grad_input_new.strides());
  return grad_input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
