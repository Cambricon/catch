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

std::set<at::ScalarType> masked_select_support_dtype{at::ScalarType::Half,
                                                     at::ScalarType::Float,
                                                     at::ScalarType::Double,
                                                     at::ScalarType::Char,
                                                     at::ScalarType::Short,
                                                     at::ScalarType::Int,
                                                     at::ScalarType::Bool,
                                                     at::ScalarType::Long};

at::Tensor cnnl_masked_select(const at::Tensor& self,
                              const at::Tensor& mask) {
  TORCH_MLU_CHECK(masked_select_support_dtype.find(self.scalar_type()) != masked_select_support_dtype.end(),
                  "masked_select mlu op not implemented for '", self.scalar_type(), "'");
  TORCH_CHECK(mask.scalar_type() != at::kByte,
    "indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool,
    "masked_select: expected BoolTensor for mask");
  at::Tensor self_tmp = self;
  at::Tensor mask_tmp = mask;
  bool isBool = (self_tmp.dtype() == at::ScalarType::Bool) ? true : false;
  // TODO(ludehui): cnnlUnarySelect kernel will raise error when input dtype is bool.
  if (isBool) {
    self_tmp = self_tmp.to(at::ScalarType::Char);
  }
  std::tie(self_tmp, mask_tmp) = at::expand_outplace(self_tmp, mask_tmp, "masked_select");
  auto memory_format = self_tmp.suggest_memory_format();
  auto output = at::empty({self_tmp.numel()}, self_tmp.options());
  auto self_contiguous = cnnl_contiguous(self_tmp, at::MemoryFormat::Contiguous);
  at::Tensor mask_contiguous = cnnl_contiguous(mask_tmp, at::MemoryFormat::Contiguous);
  cnnl_masked_select_internal(output, self_contiguous, mask_contiguous);
  if (isBool) {
    return output.to(at::ScalarType::Bool);
  }
  return output;
}

at::Tensor& cnnl_masked_select_out(at::Tensor& output,
                                   const at::Tensor& self,
                                   const at::Tensor& mask) {
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
          "masked_select(): self and result must have the same scalar type");
  auto out = cnnl_masked_select(self, mask);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(out));
  resize_impl_mlu_(getMluTensorImpl(output), out.sizes(), out.strides());
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
