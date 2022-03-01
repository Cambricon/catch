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
// bool、float32、int32、float64、int64
std::set<at::ScalarType> nonzero_support_dtype{at::ScalarType::Bool,
                                        at::ScalarType::Int,
                                        at::ScalarType::Float,
                                        at::ScalarType::Double,
                                        at::ScalarType::Long};

at::Tensor cnnl_nonzero(const at::Tensor& self) {
  TORCH_MLU_CHECK(nonzero_support_dtype.find(self.scalar_type()) != nonzero_support_dtype.end(),
                "self dtype of mlu nonzero op not implemented for ",
                self.scalar_type());
  // we can cancel this check when cnnl support zero elements Tensor
  if (self.numel() == 0) {
    return at::empty({0, self.dim()}, self.options().dtype(at::ScalarType::Long));;
  }
  at::Tensor fake;
  auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
  return cnnl_nonzero_internal(fake, self_contiguous);;
}

at::Tensor& cnnl_nonzero_out(at::Tensor& out, const at::Tensor& self) {
  TORCH_MLU_CHECK(nonzero_support_dtype.find(self.scalar_type()) != nonzero_support_dtype.end(),
                "self dtype of mlu nonzero op not implemented for ",
                self.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Long, "the datatype of out in cnnl_nonzero_out "
    "must be Long, but got ", out.scalar_type())
  // we can cancel this check when cnnl support zero elements Tensor
  if (self.numel() == 0) {
    resize_impl_mlu_(getMluTensorImpl(out), {0, self.dim()}, c10::nullopt);
    return out;
  }
  auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
  cnnl_nonzero_internal(out, self_contiguous);
  return out;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
