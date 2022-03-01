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

at::Tensor cnnl_nonzero_internal(at::Tensor &out, const at::Tensor &self) {
  // this is a layout sensitive op, but cnnl does not support strides, so we need transpose
  auto dim_num = self.dim();
  auto handle = getCurrentHandle();
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();
  // prepare workspace
  size_t sz;
  // CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  TORCH_CNNL_CHECK(cnnlGetNumTrueWorkspaceSize(handle, self_desc.desc(), &sz));
  auto index = at::empty({static_cast<long>(sz)},
  at::TensorOptions(at::ScalarType::Byte).device(at::Device(at::Device::Type::MLU)));

  // call cnnlNumTrue to count the number of nonzero elements
  auto index_impl = getMluTensorImpl(index);
  auto index_ptr = static_cast<uint32_t*>(index_impl->cnnlMalloc());
  auto device_id = self_impl->get_device();
  torch_mlu::mlu::MLUGuard guard(device_id);
  auto* allocator = dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(c10::kMLU));
  auto&& data_ptr_cls = allocator->allocate(sizeof(uint32_t), device_id);
  auto num_nonzeros_ptr = static_cast<uint32_t*>(data_ptr_cls.get());
  TORCH_CNNL_CHECK(cnnlNumTrue(handle, self_desc.desc(), self_ptr, index_ptr, num_nonzeros_ptr));

  // call cnnlWhere to output the index of nonzero elements
  auto queue = getCurrentQueue();
  uint32_t num_nonzeros = 0;
  TORCH_CNRT_CHECK(cnrtMemcpyAsync(&num_nonzeros, num_nonzeros_ptr, sizeof(uint32_t),
    queue.queue(), CNRT_MEM_TRANS_DIR_DEV2HOST));
  queue.synchronize();
  std::vector<int64_t> outshape = {num_nonzeros, dim_num};
  int64_t out_numel = out.defined() ? out.numel() : std::numeric_limits<int64_t>::max();
  at::Tensor out_new;
  if (out.defined() && num_nonzeros * dim_num <= out.numel()) {
    resize_impl_mlu_(getMluTensorImpl(out), outshape, c10::nullopt);
    out_new = out;
  } else {
    out_new = at::empty(outshape, self.options().dtype(at::ScalarType::Long));
  }
  if (dim_num == 0) {   // support scalar input
    return out_new;
  }
  auto out_new_impl = getMluTensorImpl(out_new);
  auto out_new_ptr = out_new_impl->cnnlMalloc();
  CnnlTensorDescriptor out_new_desc;
  out_new_desc.set(out_new);
  uint32_t *strides = allocateGlobalBuffer<uint32_t>(dim_num);
  for (auto i = 0; i < dim_num; ++i) {
    *(strides + i) = static_cast<uint32_t>(self_impl->stride(i));
  }

  auto&& strides_ptr_cls = allocator->allocate(dim_num * sizeof(uint32_t), device_id);
  auto strides_ptr = static_cast<uint32_t*>(strides_ptr_cls.get());
  TORCH_CNRT_CHECK(cnrtMemcpyAsync(strides_ptr, strides, sizeof(uint32_t) * dim_num,
    queue.queue(), CNRT_MEM_TRANS_DIR_HOST2DEV));
  GlobalBuffer_placeNotifier(strides);
  TORCH_CNNL_CHECK(cnnlWhere(handle, self_desc.desc(), self_ptr, strides_ptr, index_ptr,
    out_new_desc.desc(), static_cast<int*>(out_new_ptr), false));

  if (out_numel < out_new.numel()) {
    getMluTensorImpl(out)->copy_cnnl_metadata_from(out_new_impl);
    resize_impl_mlu_(getMluTensorImpl(out), out_new.sizes(), out_new.strides());
  }

  return out_new;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
