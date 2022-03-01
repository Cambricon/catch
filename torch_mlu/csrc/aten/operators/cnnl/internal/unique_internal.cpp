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

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl__unique2_internal(
    const at::Tensor &self, bool sorted, bool return_inverse,
    bool return_counts) {
  auto self_impl = getMluTensorImpl(self);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor self_desc;
  CnnlUniqueDescriptor unique_desc;
  self_desc.set(self);
  // this interface only handle the situation dim == 0
  unique_desc.set(sorted, 0, return_inverse, return_counts);
  auto self_ptr = self_impl->cnnlMalloc();

  // get workspace
  size_t get_size_out = 0;
  at::Tensor gsh_data;
  void *gsh_data_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetUniqueWorkSpace(handle, unique_desc.desc(),
                                          self_desc.desc(), &get_size_out));
  if (get_size_out != 0) {
    gsh_data = at::zeros({static_cast<long int>(get_size_out)}, self.options());
    gsh_data_ptr = getMluTensorImpl(gsh_data)->cnnlMalloc();
  }

  at::Tensor output_len;
  void *output_len_ptr = nullptr;
  output_len = at::empty({1}, self.options().dtype(at::ScalarType::Int));
  output_len_ptr = getMluTensorImpl(output_len)->cnnlMalloc();
  TORCH_CNNL_CHECK(cnnlUniqueGetOutLen(
      handle, unique_desc.desc(), self_desc.desc(), self_ptr,
      static_cast<float *>(gsh_data_ptr), static_cast<int *>(output_len_ptr)));

  auto queue = getCurrentQueue();
  int tmp = 0;
  TORCH_CNRT_CHECK(cnrtMemcpyAsync(&tmp, output_len_ptr, sizeof(int), queue.queue(),
                                   CNRT_MEM_TRANS_DIR_DEV2HOST));
  queue.synchronize();

  at::Tensor out_data;
  void *out_data_ptr = nullptr;
  out_data = at::empty({static_cast<long int>(tmp)}, self.options());
  out_data_ptr = getMluTensorImpl(out_data)->cnnlMalloc();

  at::Tensor out_data_index;
  void *out_data_index_ptr = nullptr;
  out_data_index = at::empty_like(self, self.options().dtype(at::ScalarType::Int));
  out_data_index_ptr = getMluTensorImpl(out_data_index)->cnnlMalloc();

  at::Tensor out_counts;
  void *out_counts_ptr = nullptr;
  if (return_counts) {
      out_counts = at::empty({static_cast<long int>(tmp)},
                   self.options().dtype(at::ScalarType::Int));
      out_counts_ptr = getMluTensorImpl(out_counts)->cnnlMalloc();
  }

  TORCH_CNNL_CHECK(cnnlUnique(handle, unique_desc.desc(), self_desc.desc(), self_ptr, tmp,
                   static_cast<float *>(gsh_data_ptr), out_data_ptr,
                   static_cast<int *>(out_data_index_ptr), static_cast<int*>(out_counts_ptr)));
  return std::make_tuple(out_data, out_data_index, out_counts);
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
