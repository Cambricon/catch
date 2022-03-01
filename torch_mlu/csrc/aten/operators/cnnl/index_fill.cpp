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

at::Tensor& cnnl_index_fill_(at::Tensor & self, int64_t dim,
        const at::Tensor & index, at::Scalar value) {
  dim = at::maybe_wrap_dim(dim, self);
  auto numel = index.numel();
  auto self_numel = self.numel();
  auto numel_mul = numel * self_numel;
  if (numel_mul == 0) {
    return self;
  }

  auto input_arg = at::TensorArg(self, "input", 1);
  at::ArrayRef<at::ScalarType> types = {at::ScalarType::Float,
                                        at::ScalarType::Half,
                                        at::ScalarType::Short,
                                        at::ScalarType::Int,
                                        at::ScalarType::Char,
                                        at::ScalarType::Byte};
  at::checkScalarTypes("index_fill_", input_arg, types);
  auto index_dtype = index.scalar_type();
  TORCH_MLU_CHECK(index_dtype == at::kLong || index_dtype == at::kInt,
  "Expected object of scalar type Long but got scalar type Float for argument #3 'index' in ",
  "call to _th_index_fill_");
  TORCH_MLU_CHECK(index.dim() == 1, "Index is supposed to be a vector");
  TORCH_MLU_CHECK(dim < self.dim(), "Indexing dim %d is out of bounds of tensor", dim);

  auto memory_format = c10::MemoryFormat::Contiguous;
  auto output = at::empty(self.sizes(), self.options(), memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto index_contiguous = cnnl_contiguous(index, memory_format);

  output = cnnl_index_fill_internal(output, self_contiguous, dim, index_contiguous, value);
  self.copy_(output);
  return self;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

