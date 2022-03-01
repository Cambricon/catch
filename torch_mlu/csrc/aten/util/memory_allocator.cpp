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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <string>
#include "aten/device/notifier.h"
#include "aten/util/memory_allocator.h"

namespace torch_mlu {
namespace memory {

MemoryAllocator::~MemoryAllocator() {
  // Currently, do not destroy MLU global resources manully, destroy in ~DefaultMLUAllocator
  // See NOTE [ Destruction Order of Global Variables ] in caching_allocator.cpp
  clean();
}

MemoryAllocator MemoryAllocator::allocator;

void MemoryAllocator::clean() {
  synchronize_and_free_notifier();

  // free all pageable memory
  while (!pointer_set.empty()) {
    auto iter = pointer_set.begin();
    deallocate<void>(*iter);
  }
  pointer_set.clear();

  // clear available chunk list
  available_list.clear();

  // free all pinned memory chunks
  for (auto it = pinned_chunks.begin(); it != pinned_chunks.end();) {
    HostChunk& host_chunk = it->second;
    cnrtRet_t status = cnrtFreeHost(host_chunk.ptr);
    if (status != CNRT_RET_SUCCESS) {
      TORCH_CNRT_CHECK(status);
    }
    auto cur = it;
    ++it;
    pinned_chunks.erase(cur);
  }
  pinned_chunks.clear();
}

void MemoryAllocator::placeNotifier(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex);   // std::deque is not thread safe
  placeNotifierImpl(ptr);
}

void MemoryAllocator::synchronize_and_free_notifier() {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto n : notifiers) {
    auto notifier_sptr = n.first;
    // Currently, do not destroy MLU global resources manully, destroy in ~DefaultMLUAllocator
    // See NOTE [ Destruction Order of Global Variables ] in caching_allocator.cpp
    notifier_sptr->synchronize();
    NotifierPool_Manager.give_back_notifier(notifier_sptr);
    notifiers.pop_front();
  }
}

}  // namespace memory

void freeAllGlobalBuffer() { memory::MemoryAllocator::instance().clean(); }

static void MLUCachingHostDeleter(void* ptr) {
  memory::MemoryAllocator::instance().deallocate(ptr, true);
}

struct MLUCachingHostAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    AT_ASSERT(size >= 0);
    void *ptr = nullptr;
    ptr = memory::MemoryAllocator::instance().allocate(ptr, size, true);
    return {ptr, ptr, &MLUCachingHostDeleter, at::DeviceType::CPU};
  }

  void* raw_allocate(size_t size) {
    AT_ASSERT(size >= 0);
    void *ptr = nullptr;
    ptr = memory::MemoryAllocator::instance().allocate(ptr, size, true);
    return ptr;
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &MLUCachingHostDeleter;
  }
};

MLUCachingHostAllocator mlu_caching_host_allocator;

at::Tensor pinMemory(const at::Tensor& data) {
  if (data.type().backend() != c10::Backend::CPU) {
    AT_ERROR("cannot pin '", data.type().toString(), "' only dense CPU tensors can be pinned");
  }
  auto* allocator = &mlu_caching_host_allocator;
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
        data.sizes(), data.strides(), data.dtype().itemsize()),
      allocator,
      /*resizable=*/false
);
  auto tensor = at::empty({0}, data.options()).set_(storage, 0, data.sizes(), data.strides());
  tensor.copy_(data);
  return tensor;
}

at::Tensor pinMemoryEmpty(at::IntArrayRef size,
                          const at::TensorOptions& options) {
  auto* allocator = &mlu_caching_host_allocator;
  int64_t nelements = at::prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      storage_impl, c10::DispatchKey::CPU, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  return tensor;
}

at::Allocator* getMLUCachingHostAllocator() {
  return &mlu_caching_host_allocator;
}

}  // namespace torch_mlu
