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

void cnnl_topk_internal(at::Tensor& values, at::Tensor& indices,
                        const at::Tensor& self, int64_t k,
                        int64_t dim, bool largest, bool sorted) {
  auto memory_format = self.suggest_memory_format();
  dim = modify_dim_based_on_layout(dim, memory_format);
  auto self_impl = getMluTensorImpl(self);
  auto values_impl = getMluTensorImpl(values);
  auto indices_impl = getMluTensorImpl(indices);

  // get cnnl sizes and strides
  auto self_sizes_strides = get_tensor_size_stride(self, memory_format);
  auto vaules_sizes_strides = get_tensor_size_stride(values, memory_format);
  auto indices_sizes_strides = get_tensor_size_stride(indices, memory_format);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor values_desc;
  CnnlTensorDescriptor indices_desc;

  // get cnnl descriptor
  self_desc.set(self, std::get<0>(self_sizes_strides),
                std::get<1>(self_sizes_strides), CNNL_LAYOUT_ARRAY);
  values_desc.set(values, std::get<0>(vaules_sizes_strides),
                  std::get<1>(vaules_sizes_strides), CNNL_LAYOUT_ARRAY);
  indices_desc.set(indices, std::get<0>(indices_sizes_strides),
                   std::get<1>(indices_sizes_strides), CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto self_ptr = self_impl->cnnlMalloc();
  auto values_ptr = values_impl->cnnlMalloc();
  auto indices_ptr = indices_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlTopKTensor(handle, self_desc.desc(), self_ptr,
                                  k, dim, largest, sorted, values_desc.desc(),
                                  values_ptr, indices_desc.desc(), indices_ptr));
}

std::tuple<at::Tensor, at::Tensor> cnnl_sort_internal(const at::Tensor& input,
                                                      int64_t dim,
                                                      bool largest,
                                                      bool sorted) {
  auto memory_format = input.suggest_memory_format();
  const int k_change = input.sizes()[dim];
  dim = modify_dim_based_on_layout(dim, memory_format);
  auto output = at::empty(input.sizes(), input.options(), memory_format);
  auto index = at::empty(input.sizes(),
                         input.options().dtype(caffe2::TypeMeta::Make<long>()),
                         memory_format);

  if (input.numel() == 0) {
    return std::make_tuple(output, index);
  }

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto index_impl = getMluTensorImpl(index);

  // get cnnl sizes and strides
  auto input_sizes_strides = get_tensor_size_stride(input, memory_format);
  auto output_sizes_strides = get_tensor_size_stride(output, memory_format);
  auto index_sizes_strides = get_tensor_size_stride(index, memory_format);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  CnnlTensorDescriptor desc_index;
  // get cnnl descriptor
  desc_input.set(input, std::get<0>(input_sizes_strides),
                 std::get<1>(input_sizes_strides), CNNL_LAYOUT_ARRAY);
  desc_output.set(output, std::get<0>(output_sizes_strides),
                  std::get<1>(output_sizes_strides), CNNL_LAYOUT_ARRAY);
  desc_index.set(index, std::get<0>(index_sizes_strides),
                   std::get<1>(index_sizes_strides), CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto index_ptr = index_impl->cnnlMalloc();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlTopKTensor(
      handle, desc_input.desc(), input_ptr, k_change, dim, largest, sorted,
      desc_output.desc(), output_ptr, desc_index.desc(), index_ptr));

  // to be consistent with the GPU, we need cast back to long type
  return std::make_tuple(output, index);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
