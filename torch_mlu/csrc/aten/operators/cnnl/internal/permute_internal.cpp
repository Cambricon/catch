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
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

// permute behaviour is same with cpu side.
/* example:
  contiguous tensor input with sizes (2, 3, 4, 2), strides (24 ,8 ,2 ,1);
  std::vector<int64> permute({0, 2, 3, 1});
  temp_output = at::permute(input, permute);
  output = cnnl_contiguous(temp_output, MemoryFormat);
  detail:
    temp_output is not contigous, and sizes (2, 4, 2, 3) and strides (24, 2, 1, 8);
    if u need contiguous tensor with special MemoryFormat, need using like:
    output = at::permute(input, permute).contiguous(MemoryFormat);
    Python side:
      >>> a.size() original tensor
      torch.Size([2, 3, 4, 2])
      >>> a.stride()
      (24, 8, 2, 1)
      >>> b.size() permute tensor
      torch.Size([2, 4, 2, 3])
      >>> b.stride()
      (24, 2, 1, 8)
      >>> c.size() b.contiguous()
      torch.Size([2, 4, 2, 3])
      >>> c.stride()
      (24, 6, 3, 1)
      >>> d.size() b.contiguous(memory_format=torch.channels_last)
      torch.Size([2, 4, 2, 3])
      >>> d.stride()
      (24, 1, 12, 4)
*/

at::Tensor& cnnl_permute_out_internal(at::Tensor& output,
                                     const at::Tensor& self,
                                     at::IntArrayRef dims) {
  int p_dims = self.dim();
  TORCH_MLU_CHECK(p_dims == dims.size(),
    "number of dims don't match in permute.");
  if (self.is_contiguous(c10::MemoryFormat::Contiguous) == false) {
    TORCH_MLU_CHECK(false, "Self tensor Only support channels first contiguous.");
  }
  auto permute = dims.vec();
  auto sort_permute = permute;
  std::sort(sort_permute.begin(), sort_permute.end(), std::less<int64_t>());
  if (permute == sort_permute) {
    cnnl_copy_internal(output, self);
    return output;
  }

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self);
  output_desc.set(output);

  std::vector<int> cnnl_permute(p_dims, 0);
  for (int i = 0; i < p_dims; ++i) {
    cnnl_permute[i] = static_cast<int>(permute[i]);
  }
  CnnlTransposeDescriptor trans_desc;
  trans_desc.set(p_dims, cnnl_permute.data());

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // Get workspace
  at::Tensor trans_workspace;
  size_t workspace_size = 0;
  void* workspace_ptr = nullptr;
  cnnlGetTransposeWorkspaceSize(handle, input_desc.desc(),
                                trans_desc.desc(), &workspace_size);
  if (workspace_size != 0) {
    trans_workspace = at::empty({static_cast<long>(workspace_size)},
                                self.options().dtype(at::kByte));
    auto workspace_impl = getMluTensorImpl(trans_workspace);
    workspace_ptr = workspace_impl->cnnlMalloc();
  }

  TORCH_CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc.desc(), input_desc.desc(),
                                    input_ptr, output_desc.desc(), output_ptr,
                                    workspace_ptr, workspace_size));
  return output;
}

at::Tensor cnnl_permute_internal(const at::Tensor& self,
                                  at::IntArrayRef dims) {
  int p_dims = self.dim();
  TORCH_MLU_CHECK(p_dims == dims.size(),
    "number of dims don't match in permute.");
  if (self.is_contiguous(c10::MemoryFormat::Contiguous) == false) {
    TORCH_MLU_CHECK(false, "Self tensor Only support channels first contiguous.");
  }
  auto permute = dims.vec();

  // input
  auto input_impl = getMluTensorImpl(self);

  // output
  auto input_size = self.sizes().vec();
  std::vector<int64_t> output_size(p_dims);
  for (int i = 0; i < p_dims; ++i) {
    output_size[i] = static_cast<int64_t>(input_size[permute[i]]);
  }
  // output is CF contiguous
  auto output = at::empty(output_size, self.options(), c10::MemoryFormat::Contiguous);
  return cnnl_permute_out_internal(output, self, dims);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
