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

static inline void addmm_check(const at::Tensor& self, const at::Tensor& mat1,
                               const at::Tensor& mat2) {
  TORCH_MLU_CHECK(self.scalar_type() == mat1.scalar_type(),
              "Expected object of scalar type ", self.scalar_type(),
              " but got scalar type ",  mat1.scalar_type(), " for argument #2 'mat1'");
  TORCH_MLU_CHECK(self.scalar_type() == mat2.scalar_type(),
              "Expected object of scalar type ", self.scalar_type(),
              " but got scalar type ",  mat2.scalar_type(), " for argument #3 'mat2'");
  TORCH_MLU_CHECK(at::isFloatingType(self.scalar_type()),
              "addmm on mlu only support input tensors scalar type Float");
  TORCH_MLU_CHECK(self.dim() == 2,
              "The tensor of input's dim must 2, but found ", self.dim());
  TORCH_MLU_CHECK(mat1.dim() == 2,
              "The tensor of mat1's dim must equal to input's dim, expected 2, but found ",
              mat1.dim());
  TORCH_MLU_CHECK(mat2.dim() == 2,
              "The tensor of mat2's dim must equal to input's dim, expected 2, but found ",
              mat2.dim());
  TORCH_MLU_CHECK(mat1.size(1) == mat2.size(0),
              "Size mismatch, the size of mat2's dim 0 must equal to the size of mat1's dim 1");
  TORCH_MLU_CHECK(self.size(0) == mat1.size(0),
              "The expanded size of the tensor ", mat1.size(0), " must match the existing size ",
              self.size(0), " at non-singleton dimension 0.");
  TORCH_MLU_CHECK(self.size(1) == mat2.size(1),
              "The expanded size of the tensor ", mat2.size(1), " must match the existing size ",
              self.size(1), " at non-singleton dimension 1.");
}

void AddmmImpl(const at::Tensor& self, const at::Tensor& mat1,
                const at::Tensor& mat2, at::Tensor& result0) {
  if (mat1.numel() == 0) {
    result0 = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    auto mat1_contiguous = cnnl_contiguous(mat1, mat1.suggest_memory_format());
    auto mat2_contiguous = cnnl_contiguous(mat2, mat2.suggest_memory_format());
    bool is_trans_mat1 = false;
    bool is_trans_mat2 = false;
    if (Global::instance().isUsingFloatingDevice()) {
      result0 = cnnl_mm_internal(mat1_contiguous, mat2_contiguous, 0, 0,
                                 mat1_contiguous.options(), is_trans_mat1,
                                 is_trans_mat2, /*run_fp32*/ true);
    }
  }
}

at::Tensor cnnl_addmm(const at::Tensor& self, const at::Tensor& mat1,
                      const at::Tensor& mat2, at::Scalar beta,
                      at::Scalar alpha) {
  Tensor out = at::empty({0}, self.options());
  cnnl_addmm_out(out, self, mat1, mat2, beta, alpha);
  return out;
}

at::Tensor & cnnl_addmm_(at::Tensor & self, const at::Tensor & mat1,
                         const at::Tensor & mat2, at::Scalar beta,
                         at::Scalar alpha) {
  TORCH_MLU_CHECK(self.dim() == 2,
          "matrix expected, 1D and 0 dimension tensor does not support inplace");
  cnnl_addmm_out(self, self, mat1, mat2, beta, alpha);
  return self;
}

at::Tensor & cnnl_addmm_out(at::Tensor & out, const at::Tensor & self,
                            const at::Tensor & mat1, const at::Tensor & mat2,
                            at::Scalar beta, at::Scalar alpha) {
  at::Tensor self_expand = self;
  if ((&out != &self) && (self.dim() <= 1 || self.size(0) == 1 || self.size(1) == 1)) {
    self_expand = cnnl_expand_internal(self, {mat1.size(0), mat2.size(1)}, false);
  }
  addmm_check(self_expand, mat1, mat2);
  auto self_contiguous = cnnl_contiguous(self_expand, self_expand.suggest_memory_format());
  out.resize_(self_contiguous.sizes());
  if (self.numel() == 0) {
    return out;
  }
  at::Tensor result0;
  AddmmImpl(self_expand, mat1, mat2, result0);
  auto res0_contiguous = cnnl_contiguous(result0, result0.suggest_memory_format());
  at::Tensor tmp = at::empty_like(res0_contiguous);
  cnnl_optensor_out_internal(tmp, self_contiguous, res0_contiguous,
                             beta, alpha, CNNL_OP_TENSOR_ADD);
  cnnl_copy_(out, tmp, true);
  return out;
  }
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
