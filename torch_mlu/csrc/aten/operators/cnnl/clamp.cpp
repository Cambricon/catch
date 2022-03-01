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


at::Tensor cnnl_clamp(const at::Tensor& self, at::optional<at::Scalar> min,
                      at::optional<at::Scalar> max) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output = at::empty_like(self_contiguous);
  return cnnl_clamp_internal(output, self_contiguous, min, max);
}

at::Tensor& cnnl_clamp_(at::Tensor& self, at::optional<at::Scalar> min,
                        at::optional<at::Scalar> max) {
  return cnnl_clamp_internal(self, self, min, max);
}

at::Tensor cnnl_clamp_min(const at::Tensor& self, at::Scalar min) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output = at::empty_like(self_contiguous);
  return cnnl_clamp_internal(output, self_contiguous, min, at::optional<at::Scalar>());
}

at::Tensor& cnnl_clamp_min_(at::Tensor& self, at::Scalar min) {
  return cnnl_clamp_internal(self, self, min, at::optional<at::Scalar>());
}

at::Tensor& cnnl_clamp_min_out(at::Tensor& out, const at::Tensor & self,
                              at::Scalar min) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  return cnnl_clamp_internal(out, self_contiguous, min, at::optional<at::Scalar>());
}

at::Tensor& cnnl_clamp_out(at::Tensor& output, const at::Tensor& self,
                           at::optional<at::Scalar> min,
                           at::optional<at::Scalar> max) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  if (output.numel() >= self_contiguous.numel()) {
    resize_impl_mlu_(getMluTensorImpl(output), self_contiguous.sizes(), self_contiguous.strides());
    return cnnl_clamp_internal(output, self_contiguous, min, max);
  }
  auto output_new = at::empty_like(self_contiguous);
  cnnl_clamp_internal(output_new, self_contiguous, min, max);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(output_new));
  resize_impl_mlu_(getMluTensorImpl(output), output_new.sizes(), output_new.strides());
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
