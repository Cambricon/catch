#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include "aten/core/caching_allocator.h"
#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

const int iterations = 100;
const size_t size = 10 * 1024;
const size_t free_size = 100 * 1024 * 1024;  // 100 Mibs
const size_t large_buffer_size = 36 * 1024 * 1024;  // 36 Mibs

TEST(MLUCachingAllocatorTest, allocate) {
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = allocator->allocate(size, device);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, emptyCachedMem) {
  for (int s = 0; s < iterations; ++s) {
    auto allocator = dynamic_cast<torch_mlu::MLUCachingAllocator*>(
        at::GetAllocator(at::kMLU));
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = allocator->allocate(size * size, device);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
    emptyCachedMem();
  }
}

void thread_func() {
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  int16_t device = current_device();
  for (int i = 0; i < iterations; i++) {
    auto data_ptr = allocator->allocate(size, device);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiThread) {
  for (int i = 0; i < 100; ++i) {
    std::thread t{thread_func};
    t.join();
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiDevice) {
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  for (int d = 0; d < device_count(); ++d) {
    setDevice(d);
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = allocator->allocate(size, device);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
  }
}

TEST(MLUCachingAllocatorTest, recordQueue) {
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = allocator->allocate(size, device);
    recordQueue(data_ptr, getQueueFromPool());
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, getAllocationSize) {
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  size_t malloc_size = free - free_size;
  auto data_ptr0 = allocator->allocate(malloc_size, device);
  size_t free_size0 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size0, &total));
  auto data_ptr1 = allocator->allocate(large_buffer_size, device);
  size_t free_size1 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size1, &total));
  size_t diff = free_size0 - free_size1;
  TORCH_CHECK(diff == large_buffer_size, "diff not equal large_buffer_size!");
}

}  // namespace torch_mlu
