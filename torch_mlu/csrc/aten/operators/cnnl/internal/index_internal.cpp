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
#include "aten/operators/cnnl/internal/index_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_index_internal(at::Tensor& self,
                               std::vector<at::Tensor> indices,
                               std::vector<int64_t> output_sizes) {
  // To initialize indices ptr with nullptr (for dim check in cnnl).
  std::vector<void *> indices_ptr(8);

  // To perserve CnnlTensorDescriptor
  std::vector<CnnlTensorDescriptor> desc_pool;

  // To perserve the transposed tensor
  std::vector<at::Tensor> not_skip_pool;

  int output_dim = output_sizes.size();
  std::vector<int> output_dims(output_dim);
  std::transform(output_sizes.begin(), output_sizes.end(),
                 output_dims.begin(), [](int64_t x) { return static_cast<int>(x);  });


  // To initialize cnnlTensorDescriptor_t with nullptr (for dim check in cnnl).
  std::vector<cnnlTensorDescriptor_t> indices_desc(8);
  auto self_impl = getMluTensorImpl(self);
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  auto self_ptr = self_impl->cnnlMalloc();
  auto handle = getCurrentHandle();

  for (int i = 0 ; i < indices.size(); ++i) {
    if (indices[i].defined()) {
      TORCH_MLU_CHECK(indices[i].dim() > 0, "zero dimension tensor!");
      // TODO(liuyuxin): indices transpose to NCHW, will be deprecated in future
      auto indice  = indices[i].contiguous();
      not_skip_pool.emplace_back(indice);
      auto impl = getMluTensorImpl(indice);
      desc_pool.emplace_back();
      desc_pool.back().set(indice);
      indices_ptr[i] = impl->cnnlMalloc();
      indices_desc[i] = desc_pool.back().desc();
    } else {
      indices_ptr[i] = nullptr;
      indices_desc[i] = nullptr;
    }
  }

  auto output = at::empty(output_sizes, self.options());
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);
  // prepare output dims MLU tensor
  auto output_dim_tensor = at::empty({1}, self.options().dtype(at::ScalarType::Int));
  auto output_dims_tensor = at::empty({8}, self.options().dtype(at::ScalarType::Int));
  auto output_dim_ptr = getMluTensorImpl(output_dim_tensor)->cnnlMalloc();
  auto output_dims_ptr = getMluTensorImpl(output_dims_tensor)->cnnlMalloc();

  // prepare cnnl workspace
  // For Bool AdavancedIndex, the workspace will be zero.
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetAdvancedIndexWorkspaceSize(handle, self_desc.desc(),
          indices_desc.data(), &workspace_size));
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                              self.options().dtype(at::ScalarType::Byte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  // call cnnl advancedindex interface.
  TORCH_CNNL_CHECK(cnnlAdvancedIndex(handle, self_desc.desc(), self_ptr,
                              indices_desc.data(), indices_ptr.data(),
                              workspace_ptr, workspace_size, output_desc.desc(),
                              output_ptr, output_dim_ptr,
                              output_dims_ptr));
  // add synchronization point to receive output dims.
  if (indices[0].scalar_type() == at::ScalarType::Bool) {
    auto tmp = output_dims_tensor.cpu();
    std::vector<int64_t> output_size(output_dim);
    for (int i=0; i< output_dim; i++) {
        output_size[i] = tmp[i].item().to<int64_t>();
    }
    resize_impl_mlu_(getMluTensorImpl(output), output_size, c10::nullopt);
  }
  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
