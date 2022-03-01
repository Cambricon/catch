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

at::Tensor& cnnl_index_put_internal(at::Tensor& output, const at::Tensor& self,
                                    std::vector<at::Tensor> indices, const at::Tensor& value,
                                    bool accumulate) {
  auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
  auto output_contiguous = cnnl_contiguous(output, c10::MemoryFormat::Contiguous);
  auto value_contiguous = cnnl_contiguous(value, c10::MemoryFormat::Contiguous);
  // to preserve descriptor
  std::vector<CnnlTensorDescriptor> not_skip_desc;

  // to preserve transposed tensor
  std::vector<at::Tensor> not_skip_tensor;
  std::vector<cnnlTensorDescriptor_t> indices_desc;
  std::vector<void *> indices_ptr_list;

  // initialize descriptor
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descValue;
  descInput.set(self_contiguous);
  descOutput.set(output_contiguous);
  descValue.set(value_contiguous);

  // to generate indice ptr & descriptor
  for (auto i = 0; i < indices.size(); ++i) {
    if (indices[i].defined()) {
      auto indice = cnnl_contiguous(indices[i], c10::MemoryFormat::Contiguous);
      not_skip_tensor.emplace_back(indice);
      auto impl = getMluTensorImpl(indice);
      indices_ptr_list.emplace_back(impl->cnnlMalloc());
      not_skip_desc.emplace_back();
      not_skip_desc.back().set(indice);
      indices_desc.emplace_back(not_skip_desc.back().desc());
    } else {
      indices_ptr_list.emplace_back(nullptr);
      indices_desc.emplace_back(nullptr);
    }
  }

  // malloc mlu memory
  auto input_impl = getMluTensorImpl(self_contiguous);
  auto output_impl = getMluTensorImpl(output_contiguous);
  auto value_impl = getMluTensorImpl(value_contiguous);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto value_ptr = value_impl->cnnlMalloc();
  // prepare cnnl workspace
  size_t workspace_size = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(
      cnnlGetIndexPutWorkspaceSize(handle, descInput.desc(), indices_desc.data(),
                                   indices_desc.size(), descValue.desc(), accumulate,
                                   &workspace_size));
  auto workspace = at::zeros(workspace_size,
      self_contiguous.options().dtype(at::ScalarType::Byte).device(at::Device::Type::MLU));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  for (unsigned int i = 0; i < indices.size(); ++i) {
    if (!indices[i].defined() || (indices[i].numel() == 0))
        continue;
    TORCH_CHECK(indices[i].dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
  }
  TORCH_CNNL_CHECK(cnnlIndexPut(handle, descInput.desc(), input_ptr,
                                indices_desc.data(), indices_ptr_list.data(),
                                indices_desc.size(),
                                descValue.desc(), value_ptr,
                                workspace_ptr, workspace_size, accumulate, true,
                                descOutput.desc(), output_ptr));

  if (!output.is_contiguous(c10::MemoryFormat::Contiguous)) {
    output.copy_(output_contiguous);
  }
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
