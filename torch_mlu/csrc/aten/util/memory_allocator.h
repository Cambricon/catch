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

#pragma once

#include <c10/util/Exception.h>
#include <ATen/Tensor.h>
#include <deque>
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "aten/util/common.h"
#include "aten/device/notifier.h"
#include "aten/device/caching_notifier.h"

namespace torch_mlu {
namespace memory {

struct HostChunk {
  size_t size;   // allocation size
  void*  ptr;   // host memory pointer
  bool allocated;
  int notifier_count;


  explicit HostChunk(size_t size, void* ptr = nullptr, bool allocated = false) :
      size(size), ptr(ptr), allocated(allocated), notifier_count(0) {}
};

static bool ChunkComparator(const HostChunk& a, const HostChunk& b) {
  // sort by size, break ties with pointer
  if (a.size != b.size) {
    return a.size < b.size;
  }
  return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
}

typedef bool (*Comparison)(const HostChunk&, const HostChunk&);
typedef std::unordered_map<void*, HostChunk> ChunkMap;


// Allocate one buffer
template <typename T>
inline T* allocateMemory(int64_t size) {
  T* tmp = static_cast<T*>(malloc(size * sizeof(T)));
  TORCH_CHECK(tmp != nullptr, "Fail to allocate memory!!!");
  return tmp;
}

template <>
inline void* allocateMemory(int64_t size) {
  void* tmp = static_cast<void*>(malloc(size));
  TORCH_CHECK(tmp != nullptr, "Fail to allocate memory!!!");
  return tmp;
}

// Deallocate one buffer
template <typename T>
inline void deallocateMemory(T* buffer) {
  if (buffer) {
    free(buffer);
  }
}

// A singleton class to hold global memory for bind const data
// note: when you add new public function in this class, you may
// need add a lock guard protection
class MemoryAllocator {
  // SINGLETON(MemoryAllocator);

  public:
  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;
  static MemoryAllocator& instance() {
    return allocator;
  }

  // Allocate one buffer
  template <typename T>
  T* allocate(int64_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    process_notifiers();
    auto ptr = allocateMemory<T>(size);
    pointer_set.insert(static_cast<void*>(ptr));
    return ptr;
  }

  // Allocate pinned host chunk
  void* allocate(void* ptr, int64_t size, bool is_pinned) {
    std::lock_guard<std::mutex> lock(mutex);
    AT_ASSERT(is_pinned, "Allocation must be pinned");

    process_notifiers();
    // search for the smallest block which can hold this allocation
    HostChunk search_key(size);
    auto it = available_list.lower_bound(search_key);
    if (it != available_list.end()) {
      HostChunk& chunk = pinned_chunks.at(it->ptr);
      AT_ASSERT(!chunk.allocated && chunk.notifier_count == 0,
              "Host chunk allocation requires free and no notifier count!");
      chunk.allocated = true;
      ptr = chunk.ptr;
      available_list.erase(it);
      return ptr;
    }
    TORCH_CNRT_CHECK(cnrtHostMalloc(&ptr, size));
    pinned_chunks.insert({ptr, HostChunk(size, ptr, true)});
    return ptr;
  }

  // Free the specified buffer
  template <typename T>
  void deallocate(T* buffer) {
    if (pointer_set.find(static_cast<void*>(buffer)) != pointer_set.end()) {
      deallocateMemory<T>(buffer);
      pointer_set.erase(static_cast<void*>(buffer));
    }
  }

  // Free the pinned host chunk
  void deallocate(void* ptr, bool is_pinned) {
    std::lock_guard<std::mutex> lock(mutex);
    AT_ASSERT(is_pinned, "Deallocation must be pinned");
    process_notifiers();
    auto it = pinned_chunks.find(static_cast<void*>(ptr));
    if (it != pinned_chunks.end()) {
      HostChunk& chunk = it->second;
      AT_ASSERT(chunk.allocated, "The chunk should be allocated!");
      chunk.allocated = false;
      // insert notify to check whether this cpu memory is using.
      placeNotifierImpl(static_cast<void*>(ptr));
      chunk.notifier_count++;
    } else {
      LOG(ERROR) << "The host chunk is not in MLUCachingHostAllocator!";
    }
  }

  // Free all the buffer in pointer_set
  void clean();

  // place a notifier tag with the current corresponding pointer
  void placeNotifier(void* ptr);

  /*!
   * @brief Synchronize the task and wait for the marker to complete before releasing
   * all data pointers.
   *
   * @param[in] destroy_notifier
   *   Specifies whether to synchronize and release the notifier.
   *   When allocator clears the cache, all notifiers need to be synchronized and
   *   released. Otherwise, notifier need not be synchronized and released
   *   when allocator is released.
   */
  void synchronize_and_free_notifier();

  // check if the ptr is pinned
  template<typename T>
  inline bool isPinnedPtr(T* ptr) {
    auto iter = pinned_chunks.find(ptr);
    if (iter != pinned_chunks.end()) {
        return true;
    } else {
        return false;
    }
  }

 private:
  // default init
  MemoryAllocator() : available_list(ChunkComparator) {}
  ~MemoryAllocator();

  static MemoryAllocator allocator;

  std::mutex mutex;

  std::unordered_set<void*> pointer_set;

  ChunkMap pinned_chunks;

  std::set<HostChunk, Comparison> available_list;

  // // outstanding mlu notifiers
  std::deque<std::pair<std::shared_ptr<Notifier>, void*>> notifiers;

  inline void deallocateOp(void* ptr) {
    auto iter = pinned_chunks.find(ptr);
    if (iter != pinned_chunks.end()) {
      HostChunk& chunk = iter->second;
      chunk.notifier_count--;
      if (chunk.notifier_count == 0 && !chunk.allocated) {
        available_list.insert(chunk);
      }
    } else {
      deallocate<void>(ptr);
    }
  }

  // Process outstanding notifiers.
  // When the marked notifier is completed the notifier is removed and
  // the corresponding cpu pointer is released.
  void process_notifiers() {
    while (!notifiers.empty()) {
      auto& n = notifiers.front();
      auto notifier_sptr = n.first;
      auto ptr = n.second;
      torch_mlu::mlu::MLUGuard guard(notifier_sptr->device_index());
      const bool ret = notifier_sptr->query();
      if (ret == false) {
        break;
      }
      deallocateOp(ptr);
      NotifierPool_Manager.give_back_notifier(notifier_sptr);
      notifiers.pop_front();
    }
  }

  // implementation of place a notifier tag with the current
  // corresponding pointer
  inline void placeNotifierImpl(void* ptr) {
    auto queue = getCurrentQueue();
    c10::DeviceIndex device_id = static_cast<c10::DeviceIndex>(queue.device_index());
    auto notifier_sptr = NotifierPool_Manager.alloc_notifier(device_id);
    notifier_sptr->place(queue);
    notifiers.emplace_back(notifier_sptr, ptr);
  }
};

}  // namespace memory

// This function is designed to allocate buffers
template <typename T>
T* allocateGlobalBuffer(int64_t size) {
  return memory::MemoryAllocator::instance().allocate<T>(size);
}

// This function is designed to free buffer
template <typename T>
void freeGlobalBuffer(T* buffer) {
  memory::MemoryAllocator::instance().deallocate<T>(buffer);
}

// This function marks the specified buffer and frees it when the query mark
// status is complete.
template <typename T>
void GlobalBuffer_placeNotifier(T* buffer) {
  memory::MemoryAllocator::instance().placeNotifier(buffer);
}

// This function is designed to free all buffers
void freeAllGlobalBuffer();

// To allocate pinned memory for data
at::Tensor pinMemory(const at::Tensor& data);

// To allocate pinned memory for empty tensor
at::Tensor pinMemoryEmpty(at::IntArrayRef size, const at::TensorOptions& options);

// To check if the memory ptr is pinned or not
template <typename T>
bool isPinned(T* ptr) {
  return memory::MemoryAllocator::instance().isPinnedPtr<T>(ptr);
}

// To get MLUCachingHostAllocator
at::Allocator* getMLUCachingHostAllocator();

}  // namespace torch_mlu
