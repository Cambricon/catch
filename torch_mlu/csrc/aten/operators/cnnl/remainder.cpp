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
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_remainder(const at::Tensor& self, const at::Tensor& other) {
  // get output size
  auto output_size = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_size, self.options());
  return cnnl_remainder_internal(output, self, other);
}

at::Tensor cnnl_remainder(const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(self.sizes(), other, self.options());
  return cnnl_remainder(self, other_tensor);
}

at::Tensor& cnnl_remainder_(at::Tensor& self, const at::Tensor& other) {
  auto output_size = at::infer_size(self.sizes(), other.sizes());
  TORCH_MLU_CHECK(
      output_size == self.sizes(),
      "output with shape ",
      self.sizes(),
      " doesn't match the broadcast shape [",
      output_size,
      "]");
  return cnnl_remainder_internal(self, self, other);
}

at::Tensor& cnnl_remainder_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(self.sizes(), other, self.options());
  return cnnl_remainder_(self, other_tensor);
}

at::Tensor& cnnl_remainder_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto output_size = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_size, self.options());
  if (out.numel() >= output.numel()) {
    resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
    return cnnl_remainder_internal(out, self, other);
  }
  cnnl_remainder_internal(output, self, other);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_remainder_out(
    at::Tensor& out,
    const at::Tensor& self,
    at::Scalar other) {
  auto other_tensor = at::full(self.sizes(), other, self.options());
  return cnnl_remainder_out(out, self, other_tensor);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
