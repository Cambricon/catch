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

at::Tensor cnnl_bmm_internal(const at::Tensor &self, const at::Tensor &other,
                             const int self_position, const int other_position,
                             at::TensorOptions self_options, bool is_trans_self,
                             bool is_trans_other, bool run_fp32) {
  // get the shape of output
  auto output = getBatchmatmulOut(self, other, is_trans_self, is_trans_other, self_options);
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
    TORCH_CNNL_CHECK(
        cnnlSetTensorDescriptorPosition(desc_self.desc(), self_position));
    TORCH_CNNL_CHECK(
        cnnlSetTensorDescriptorPosition(desc_other.desc(), other_position));
  }
  // get current handle
  auto handle = getCurrentHandle();

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetBatchMatMulBCastWorkspaceSize(handle,
                                                        desc_self.desc(),
                                                        desc_other.desc(),
                                                        desc_output.desc(),
                                                        &workspace_size));
  auto workspace = at::empty(workspace_size, self.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  // get the mlu ptr
  auto self_ptr = self_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // compute ops
  TORCH_CNNL_CHECK(cnnlBatchMatMulBCast(
      /* handle         */ handle,
      /* is_trans_a     */ is_trans_self,
      /* is_trans_b     */ is_trans_other,
      /* a_desc         */ desc_self.desc(),
      /* a              */ self_ptr,
      /* b_desc         */ desc_other.desc(),
      /* b              */ other_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* c_desc         */ desc_output.desc(),
      /* c              */ output_ptr));

  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
