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

#include "aten/device/queue.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include "aten/util/python_interface.h"

namespace torch_mlu {

/**
 * MLUQueueInternals is an abstraction for cnrtQueue_t to manage the creation
 * and destruction of cnrtQueue_t.
 */
struct MLUQueueInternals {
  MLUQueueInternals() = default;
  // Currently, do not destroy MLU global resources manully, destroy in ~DefaultMLUAllocator
  // See NOTE [ Destruction Order of Global Variables ] in caching_allocator.cpp
  ~MLUQueueInternals() {}
  void create(int index, int id = -1) {
    if (queue) return;
    queue_id = id;
    device_index = index;
    auto cur_device = current_device();
    setDevice(index);
    // The creation of the corresponding device queue needs to be consistent
    // with maintaining the current device.
    TORCH_CNRT_CHECK(cnrtQueueCreate(&queue));
    setDevice(cur_device);
  }

  DeviceIndex device_index = -1;
  int32_t queue_id = -1;
  cnrtQueue_t queue = nullptr;
};

static const int kQueuesPerPool = 32;

static std::mutex mutex;

static thread_local std::once_flag init_flag[MLU_DEVICE_NUM_MAX];

static MLUQueueInternals default_queues[MLU_DEVICE_NUM_MAX];

static std::array<MLUQueueInternals, kQueuesPerPool>
    queues_pool[MLU_DEVICE_NUM_MAX];

static std::atomic<QueueIndex> queues_pool_counters[MLU_DEVICE_NUM_MAX];

static thread_local std::vector<MLUQueueInternals> current_queues(MLU_DEVICE_NUM_MAX);

static std::once_flag device_flags[MLU_DEVICE_NUM_MAX];

static inline void check_mlu(c10::DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < device_count(),
              "The running of the task requires MLU device initialization.");
}

static QueueIndex get_idx(std::atomic<QueueIndex>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kQueuesPerPool;
}

/**
 * Thr reciprocal transformation of queue and MLUQueueInternals
 */
Queue mluQueueFromInternals(const MLUQueueInternals* ptr) {
  return Queue(ptr->queue, ptr->device_index, ptr->queue_id);
}

MLUQueueInternals* mluQueueInternals(Queue queue) {
  QueueIndex id = queue.id();
  DeviceIndex device_index = queue.device_index();
  if (id == -1) {
    return &default_queues[device_index];
  }
  return &queues_pool[device_index][id];
}

/**
 * init the sequence of the MLU queue.
 */
static void initMLUQueue(DeviceIndex device_index) {
  if (current_queues[device_index].queue) return;
  auto device_num = device_count();

  if ((device_index != -1) && (device_index < device_num)) {
    default_queues[device_index].create(device_index);
    current_queues[device_index] = default_queues[device_index];
  } else {
    for (int device_index = 0; device_index < device_num; device_index++) {
      default_queues[device_index].create(device_index);
      current_queues[device_index] = default_queues[device_index];
    }
  }
}

static void initMLUQueuePool(DeviceIndex device_index) {
  queues_pool_counters[device_index] = 0;
  for (auto i = decltype(kQueuesPerPool){0}; i < kQueuesPerPool; ++i) {
    queues_pool[device_index][i].create(device_index, i);
  }
}

Queue getQueueFromPool(DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  std::call_once(device_flags[device_index], initMLUQueuePool, device_index);
  check_mlu(device_index);
  const auto idx = get_idx(queues_pool_counters[device_index]);
  return mluQueueFromInternals(&queues_pool[device_index][idx]);
}

Queue getDefaultQueue(DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  std::call_once(init_flag[device_index], initMLUQueue, device_index);
  check_mlu(device_index);
  return mluQueueFromInternals(&default_queues[device_index]);
}

Queue getCurrentQueue(DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }
  std::call_once(init_flag[device_index], initMLUQueue, device_index);
  check_mlu(device_index);
  return mluQueueFromInternals(&current_queues[device_index]);
}

void setCurrentQueue(Queue queue) {
  auto ptr = mluQueueInternals(queue);
  DeviceIndex device_index = ptr->device_index;
  std::call_once(init_flag[device_index], initMLUQueue, device_index);
  AT_ASSERT(ptr);
  current_queues[device_index] = *ptr;
}

cnrtQueue_t getCurQueue(DeviceIndex device_index) {
  auto queue = getCurrentQueue(device_index);
  return queue.queue();
}

std::ostream& operator<<(std::ostream& stream, const Queue& queue) {
    return stream << "queue: " << queue.id() << " on device " << queue.device_index() << ". ";
}

}  // namespace torch_mlu
