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
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::set<at::ScalarType> mm_support_dtype{at::ScalarType::Half,
                                          at::ScalarType::Float,
                                          at::ScalarType::Double,
                                          at::ScalarType::Int,
                                          at::ScalarType::Char,
                                          at::ScalarType::Short};

at::Tensor cnnl_mm_internal(const at::Tensor &self, const at::Tensor &other,
                            const int self_position, const int other_position,
                            at::TensorOptions self_options, bool is_trans_self,
                            bool is_trans_other, bool run_fp32) {
  // get the shape of output
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "dimension not support");

  TORCH_MLU_CHECK(mm_support_dtype.find(self.scalar_type()) != mm_support_dtype.end(),
                  "MM mlu op not implemented for dtype of input1: '",
                  self.dtype().name(), "'");

  TORCH_MLU_CHECK(mm_support_dtype.find(other.scalar_type()) != mm_support_dtype.end(),
                  "MM mlu op not implemented for dtype of input2: '",
                  other.dtype().name(), "'");

  auto output = getMatmulOut(self, other, is_trans_self, is_trans_other, self_options);
  if (output.numel() == 0)
      return output;
  if (output.dtype().name() != std::string("float")
      && output.dtype().name() != std::string("c10::Half")) {
    CNLOG(INFO) << "Output dtype : "
                << output.dtype().name()
                << " is not supported, and will cast to float due to limit of MM op.";
    output = getMatmulOut(self, other, is_trans_self, is_trans_other,
                          self_options.dtype(at::ScalarType::Float));
  }
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // create the desc
  CnnlTensorDescriptor desc_self;
  CnnlTensorDescriptor desc_other;
  CnnlTensorDescriptor desc_output;
  auto self_type = self.dtype();
  auto other_type = other.dtype();
  if (self_type.name() == std::string("int"))
    desc_self.set(self, CNNL_DTYPE_INT31);
  else
    desc_self.set(self);

  if (other_type.name() == std::string("int"))
    desc_other.set(other, CNNL_DTYPE_INT31);
  else
    desc_other.set(other);
  desc_output.set(output);
  if (!run_fp32) {
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPosition(desc_self.desc(), self_position));
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPosition(desc_other.desc(), other_position));
  }

  auto handle = getCurrentHandle();
  auto self_ptr = self_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  float alpha_float = 1.0;
  float beta_float = 0.0;
  const void * alpha = static_cast<void *>(&alpha_float);
  const void * beta = static_cast<void *>(&beta_float);
  TORCH_CNNL_CHECK(cnnlMatMul(
      /* handle     */ handle,
      /* is_trans_a */ is_trans_self,
      /* is_trans_b */ is_trans_other,
      /* alpha      */ alpha,
      /* a_desc     */ desc_self.desc(),
      /* a          */ self_ptr,
      /* b_desc     */ desc_other.desc(),
      /* b          */ other_ptr,
      /* beta       */ beta,
      /* c_desc     */ desc_output.desc(),
      /* c          */ output_ptr));

  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
