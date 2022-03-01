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

std::set<at::ScalarType> bce_with_logits_support_dtype{
                                              at::ScalarType::Float};

at::Tensor cnnl_binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    const at::Tensor& pos_weight, int64_t reduction) {
  TORCH_MLU_CHECK(bce_with_logits_support_dtype.find(self.scalar_type()) !=
                  bce_with_logits_support_dtype.end(),
                  "binary_cross_entropy_with_logits not implemented for '",
                  self.scalar_type(), "'");
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  at::Tensor weight_contiguous = weight;
  at::Tensor pos_weight_contiguous = pos_weight;
  if (weight.defined()) {
    weight_contiguous = cnnl_contiguous(weight, memory_format);
  }
  if (pos_weight.defined()) {
    pos_weight_contiguous = cnnl_contiguous(pos_weight, memory_format);
  }
  return cnnl_bce_with_logits_internal(self_contiguous,
                                       target_contiguous,
                                       weight_contiguous,
                                       pos_weight_contiguous,
                                       reduction);
}

at::Tensor cnnl_binary_cross_entropy_with_logits_backward(const at::Tensor& grad_output,
                                                          const at::Tensor& self,
                                                          const at::Tensor& target,
                                                          const at::Tensor& weight,
                                                          const at::Tensor& pos_weight,
                                                          int64_t reduction) {
  TORCH_MLU_CHECK(bce_with_logits_support_dtype.find(self.scalar_type()) !=
                  bce_with_logits_support_dtype.end(),
                  "binary_cross_entropy_with_logits not implemented for '",
                  self.scalar_type(), "'");
  auto memory_format = self.suggest_memory_format();
  auto  grad_output_contiguous =
      cnnl_contiguous(grad_output, infer_memory_format(grad_output.dim(), memory_format));

  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto target_contiguous = cnnl_contiguous(target, memory_format);
  at::Tensor weight_contiguous = weight;
  at::Tensor pos_weight_contiguous = pos_weight;
  if (weight.defined()) {
    weight_contiguous = cnnl_contiguous(weight, memory_format);
  }
  if (pos_weight.defined()) {
    pos_weight_contiguous = cnnl_contiguous(pos_weight, memory_format);
  }
  return cnnl_bce_with_logits_bp_internal(grad_output_contiguous,
                                          self_contiguous,
                                          target_contiguous,
                                          weight_contiguous,
                                          pos_weight_contiguous,
                                          reduction);
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
