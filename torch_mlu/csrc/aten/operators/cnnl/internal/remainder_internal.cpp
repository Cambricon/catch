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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor & cnnl_remainder_internal(at::Tensor & output,
                                     const at::Tensor& self,
                                     const at::Tensor& other) {
  auto self_type = self.scalar_type();
  auto other_type = other.scalar_type();
  TORCH_MLU_CHECK(
      self_type == at::ScalarType::Double ||
      self_type == at::ScalarType::Float ||
      self_type == at::ScalarType::Half ||
      self_type == at::ScalarType::Long ||
      self_type == at::ScalarType::Int,
      "cnnl_remainder is not implemented for ",
      self_type);

  TORCH_MLU_CHECK(
      self_type == other_type, "input1 should have the same type as input2");

  TORCH_MLU_CHECK(
      self.scalar_type() == output.scalar_type(),
      "input should have the same type as output");

  if (self.numel() == 0) {
    return output;
  }

  at::Tensor self_tmp = self;
  at::Tensor other_tmp = other;
  auto shape_broadcast = broadcast_shape(self, other);
  if (self.dim() > other.dim()) {
    other_tmp = other_tmp.reshape(std::get<1>(shape_broadcast));
  } else if (self.dim() < other.dim()) {
    self_tmp = self_tmp.reshape(std::get<0>(shape_broadcast));
  }

  auto self_contiguous = cnnl_contiguous(self_tmp, c10::MemoryFormat::Contiguous);
  auto other_contiguous = cnnl_contiguous(other_tmp, c10::MemoryFormat::Contiguous);
  auto output_contiguous = cnnl_contiguous(output, c10::MemoryFormat::Contiguous);

  // get tensor impl
  auto self_impl = getMluTensorImpl(self_contiguous);
  auto other_impl = getMluTensorImpl(other_contiguous);
  auto output_impl = getMluTensorImpl(output_contiguous);

  // get current handle
  auto handle = getCurrentHandle();

  // create the desc
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor output_desc;
  self_desc.set(self_contiguous);
  other_desc.set(other_contiguous);
  output_desc.set(output_contiguous);

  // get the size of workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFloorModWorkspaceSize(
      handle, self_desc.desc(), other_desc.desc(), output_desc.desc(), &space_size));

  auto workspace = at::empty(space_size, self_contiguous.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);

  // get the mlu ptr
  auto self_ptr = self_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  // compute ops
  TORCH_CNNL_CHECK(cnnlFloorMod(handle,
                                self_desc.desc(),
                                self_ptr,
                                other_desc.desc(),
                                other_ptr,
                                output_desc.desc(),
                                output_ptr,
                                workspace_ptr,
                                space_size));
  if (!output.is_contiguous(c10::MemoryFormat::Contiguous)) {
    output.copy_(output_contiguous);
  }
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
