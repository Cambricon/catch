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
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_bitwise_op_out_internal(at::Tensor& out,
                                        const at::Tensor& self,
                                        const at::Tensor& other,
                                        const cnnlBitComputeOp_t& op_type) {
  std::vector<at::ScalarType> vec_type = {at::kBool, at::kByte, at::kChar, at::kShort,
                                          at::kInt, at::kLong};
  auto self_dtype = self.scalar_type();
  auto other_dtype = other.scalar_type();
  TORCH_MLU_CHECK(find(vec_type.begin(), vec_type.end(), self_dtype) != vec_type.end() &&
                  find(vec_type.begin(), vec_type.end(), other_dtype) != vec_type.end(),
                  "self and other only support int related types");
  auto input1_impl = getMluTensorImpl(self);
  auto input2_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(out);
  auto input1_ptr = input1_impl->cnnlMalloc();
  auto input2_ptr = input2_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  CnnlTensorDescriptor input1_desc, input2_desc, output_desc;
  auto suggest_self_layout = suggest_cnnl_layout(self);
  input1_desc.set(self, suggest_self_layout);
  input2_desc.set(other, suggest_self_layout);
  output_desc.set(out, suggest_self_layout);
  auto handle = getCurrentHandle();

  // prepare workspace
  size_t workspace_size = 0;
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetBitComputeWorkspaceSize(
      handle, input1_desc.desc(), input2_desc.desc(), output_desc.desc(),
      &workspace_size));
  workspace = at::empty(workspace_size,
                        self.options().dtype(at::ScalarType::Char));
  workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();

  TORCH_CNNL_CHECK(cnnlBitCompute_v2(handle, op_type, input1_desc.desc(), input1_ptr,
    input2_desc.desc(), input2_ptr, output_desc.desc(), output_ptr, workspace_ptr, workspace_size));
  return out;
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
