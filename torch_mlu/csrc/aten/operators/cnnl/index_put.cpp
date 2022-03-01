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
#include "aten/operators/cnnl/internal/index_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl__index_put_impl_(at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate, bool unsafe) {
  if (values.numel() == 0 ||
      (self.dim() == 1 && indices[0].numel() == 0) ||
      self.numel() == 0) {
    return self;
  }
  std::vector<at::Tensor> indices_expand;
  TORCH_MLU_CHECK(indices.size() <= self.dim(), "indices have more indices than self dim");
  TORCH_MLU_CHECK(self.scalar_type() == values.scalar_type(), \
                  "self and values must have same dtype");
  TORCH_MLU_CHECK(!indices.empty(), "indices can't be empty");
  if (indices.size() == 1 && indices[0].defined() &&
      indices[0].scalar_type() == at::ScalarType::Bool) {
    indices_expand.emplace_back(indices[0]);
  } else {
    std::tie(self, indices_expand) = make_info(self, indices);
  }
  for (const auto& indice : indices_expand) {
    // only support long, int32 and bool
    if (!indice.defined()) {
        continue;
    }
    TORCH_MLU_CHECK(indice.scalar_type() == at::ScalarType::Int ||
                    indice.scalar_type() == at::ScalarType::Bool ||
                    indice.scalar_type() == at::ScalarType::Long,
                     "support only int, bool and long");
  }
  return cnnl_index_put_internal(self, self, indices_expand, values, accumulate);
}

at::Tensor& cnnl_index_put_(at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate) {
    return cnnl__index_put_impl_(self, indices, values, accumulate, false);
}

at::Tensor cnnl_index_put(const at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate) {
    auto self_clone = self.clone(at::MemoryFormat::Preserve);
    return cnnl_index_put_(self_clone, indices, values, accumulate);
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
