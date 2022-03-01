#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "aten/util/memory_allocator.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

const int iterations = 1000;
const size_t size = 10 * 1024 * 1024;

TEST(MemoryAllocatorTest, allocateGlobalBuffer) {
  float* buffer = allocateGlobalBuffer<float>(size);
  CHECK_NE(buffer, nullptr);
  memset(buffer, 0, size);
  freeGlobalBuffer(buffer);
}

TEST(MemoryAllocatorTest, GlobalBuffer) {
  for (int i = 0; i < iterations; ++i) {
    void* buffer = static_cast<void*>(allocateGlobalBuffer<float>(size));
  }
  freeAllGlobalBuffer();
}

TEST(MemoryAllocatorTest, GlobalBufferVoid) {
  for (int i = 0; i < iterations; ++i) {
    void* buffer = allocateGlobalBuffer<void>(size);
  }
  freeAllGlobalBuffer();
}

TEST(MemoryAllocatorTest, GlobalBuffer_placeNotifier) {
  for (int i = 0; i < iterations; ++i) {
    void* buffer = static_cast<void*>(allocateGlobalBuffer<float>(size));
    GlobalBuffer_placeNotifier(buffer);
  }
  freeAllGlobalBuffer();
}

void malloc_func() {
  for (int i = 0; i < iterations; i++) {
    void* buffer = static_cast<void*>(allocateGlobalBuffer<float>(size));
    GlobalBuffer_placeNotifier(buffer);
  }
}

TEST(MemoryAllocatorTest, MultiThreadAllocateGlobalBuffer) {
  std::thread threads[100];
  for (int i = 0; i < 100; ++i) {
    threads[i] = std::thread(malloc_func);
  }
  for (auto &t : threads) {
    t.join();
  }
  freeAllGlobalBuffer();
}

}  // namespace torch_mlu
