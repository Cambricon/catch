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

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_mm(const at::Tensor &self, const at::Tensor &other) {
  auto mm_arg1 = at::TensorArg(self, "mat1", 1);
  auto mm_arg2 = at::TensorArg(other, "mat2", 1);
  at::checkDim("mm", mm_arg1, 2);
  at::checkDim("mm", mm_arg2, 2);
  TORCH_MLU_CHECK(self.size(1) == other.size(0),
     "size mismatch, m1: ", self.sizes(), ", m2: ", other.sizes(),
     " while checking arguments for mm");
  // case1: self's col and other's row are 0, return zero tensor, size like mm.
  // case2: self's row or other's col is 0, return empty tensor, size like mm.
  // case2 has a higher priority than case1.
  if (self.numel() == 0 || other.numel() == 0) {
    return at::zeros({self.size(0), other.size(1)}, self.options());
  } else {
    at::Tensor self_contiguous;
    at::Tensor other_contiguous;
    bool is_trans_self;
    bool is_trans_other;
    std::tie(self_contiguous, is_trans_self) = getMMInput(self);
    std::tie(other_contiguous, is_trans_other) = getMMInput(other);

    if (Global::instance().isUsingFloatingDevice()) {
      return cnnl_mm_internal(self_contiguous, other_contiguous, 0, 0,
                              self_contiguous.options(), is_trans_self,
                              is_trans_other, /*run_fp32*/ true);
    }
  }
}

at::Tensor cnnl_mm(const at::Tensor &self, const int self_position,
                   const at::Tensor &other, const int other_position,
                   bool is_trans_self, bool is_trans_other) {
  return cnnl_mm_internal(self, other, self_position, other_position,
                          self.options().dtype(at::ScalarType::Float),
                          is_trans_self, is_trans_other);
}

at::Tensor& cnnl_mm_out(at::Tensor& out, const at::Tensor& self,
        const at::Tensor& other) {
  auto output = cnnl_mm(self, other);
  out.resize_(output.sizes());
  out.copy_(output);
  return out;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
