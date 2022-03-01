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

at::Tensor cnnl_cat_internal(at::TensorList tensors, int64_t dim) {
  // preprocess & prepare cnnl inputs
  // legacy_cat_wrap_dim will deal with dim < 0, and will prevent below
  // TORCH_MLU_CHECK, and python exception code can't run into
  dim = legacy_cat_wrap_dim(dim, tensors);
  TORCH_MLU_CHECK(dim >= 0, "dim must be not less than 0");
  auto memory_format = tensors[0].suggest_memory_format();
  int64_t concated_dim = 0;
  at::IntList output_shape;
  std::vector<CnnlTensorDescriptor> not_skip_desc;
  std::vector<void *> input_ptr_list;
  std::vector<cnnlTensorDescriptor_t> inputs_desc;
  auto input_layout = suggest_cnnl_layout(tensors[0]);
  for (auto i = 0; i < tensors.size(); ++i) {
    if (!tensors[i].defined() ||
        (tensors[i].numel() == 0 && tensors[i].dim() == 1)) {
      continue;
    }
    TORCH_MLU_CHECK(tensors[i].dim() > 0, "zero-dimensional tensor (at position ", i,
                    ") cannot be concatenated");
    auto impl = getMluTensorImpl(tensors[i]);
    input_ptr_list.emplace_back(impl->cnnlMalloc());
    // auto input_layout = suggest_cnnl_layout(tensors[i]);
    cnnlDataType_t input_data_type = impl->getCnnlType();
    not_skip_desc.emplace_back();
    not_skip_desc.back().set(tensors[i], input_layout, input_data_type);
    inputs_desc.emplace_back(not_skip_desc.back().desc());
    concated_dim += tensors[i].size(dim);
    if (output_shape.size() == 0) output_shape = tensors[i].sizes();
  }
  if (inputs_desc.size() == 0) return at::empty({0}, tensors[0].options());

  // prepare cnnl output
  auto output_shape_vec = output_shape.vec();
  output_shape_vec[dim] = concated_dim;
  auto output = at::empty(output_shape_vec, tensors[0].options(), memory_format);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  auto output_layout = suggest_cnnl_layout(output);
  cnnlDataType_t output_data_type = output_impl->getCnnlType();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, output_layout, output_data_type);

  // prepare cnnl workspace
  size_t workspace_size = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(
      cnnlGetConcatWorkspaceSize(handle, inputs_desc.size(), &workspace_size));
  auto workspace = at::empty(at::IntList(static_cast<int64_t>(workspace_size)),
                             tensors[0].options().dtype(at::ScalarType::Byte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  // call cnnl embedding interface
  int64_t channels_last_dim = modify_dim_based_on_layout(dim, memory_format);
  TORCH_CNNL_CHECK(cnnlConcat(handle, static_cast<int>(inputs_desc.size()),
                              channels_last_dim, inputs_desc.data(),
                              input_ptr_list.data(), workspace_ptr,
                              workspace_size, output_desc.desc(),
                              output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
