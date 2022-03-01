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

static const std::set<at::ScalarType> fill_support_dtype{ at::ScalarType::Int,
                                                          at::ScalarType::Half,
                                                          at::ScalarType::Float,
                                                          at::ScalarType::Short,
                                                          at::ScalarType::Char,
                                                          at::ScalarType::Byte,
                                                          at::ScalarType::Long,
                                                          at::ScalarType::Bool,
                                                          at::ScalarType::Double };

at::Tensor& cnnl_fill_(at::Tensor& self, const at::Tensor& other) {
  TORCH_MLU_CHECK(
      other.dim() == 0,
      "fill_ only supports 0-dimension value tensor but got tensor with ",
      other.dim(), " dimensions.");
  cnnl_fill_(self, other.cpu().to(at::kFloat).item());
  return self;
}

at::Tensor& cnnl_fill_(at::Tensor& self, at::Scalar other) {
  TORCH_MLU_CHECK(fill_support_dtype.find(self.scalar_type()) != fill_support_dtype.end(),
                  "fill mlu op not implemented for '", self.scalar_type(), "'");
  if (self.numel() == 0) return self;
  auto value = other.to<float>();
  cnnl_fill_internal(self, value);
  return self;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
