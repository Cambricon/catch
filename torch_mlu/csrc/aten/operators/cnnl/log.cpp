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

at::Tensor& cnnl_log_out(at::Tensor &out, const at::Tensor &self) {
  TORCH_CHECK(self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf,
              "log not implemented for ", self.scalar_type());
  TORCH_CHECK(self.scalar_type() == out.scalar_type(),
              "log expected self dtype ", self.scalar_type(),
              " match with out dtype ", out.scalar_type());
  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  if (out.numel() >= self.numel()) {
    resize_impl_mlu_(getMluTensorImpl(out), self_contiguous.sizes(), self_contiguous.strides());
    return cnnl_log_internal(out, self_contiguous, CNNL_LOG_E);
  }
  auto output = at::empty_like(self_contiguous);
  cnnl_log_internal(output, self_contiguous, CNNL_LOG_E);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_log2_out(at::Tensor &out, const at::Tensor &self) {
  TORCH_MLU_CHECK(
      at::isFloatingType(self.scalar_type()),
      "log2 only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_MLU_CHECK(self.scalar_type() == out.scalar_type(),
                  "log2 expected dtype ", self.scalar_type(),
                  " but found ", out.scalar_type());
  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  if (out.numel() >= self.numel()) {
    resize_impl_mlu_(getMluTensorImpl(out), self_contiguous.sizes(), self_contiguous.strides());
    return cnnl_log_internal(out, self_contiguous, CNNL_LOG_2);
  }
  auto output = at::empty_like(self_contiguous);
  cnnl_log_internal(output, self_contiguous, CNNL_LOG_2);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_log10_out(at::Tensor &out, const at::Tensor &self) {
  TORCH_MLU_CHECK(
      at::isFloatingType(self.scalar_type()),
      "log10 only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_MLU_CHECK(self.scalar_type() == out.scalar_type(),
                  "log10 expected dtype ", self.scalar_type(),
                  " but found ", out.scalar_type());
  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  if (out.numel() >= self.numel()) {
    resize_impl_mlu_(getMluTensorImpl(out), self_contiguous.sizes(), self_contiguous.strides());
    return cnnl_log_internal(out, self_contiguous, CNNL_LOG_10);
  }
  auto output = at::empty_like(self_contiguous);
  cnnl_log_internal(output, self_contiguous, CNNL_LOG_10);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
