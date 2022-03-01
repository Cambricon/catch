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

#include "aten/device/device.h"
#include "aten/util/cnlog.h"
#include "aten/device/mlu_guard.h"
#include "cnrt.h"  // NOLINT

using c10::DeviceType;
using c10::DeviceIndex;
using c10::Device;
using QueueIndex = int16_t;

namespace torch_mlu {

class Queue;

/**
 * Get a Queue from the original completed sequence of the queue.
 * Queue Pool creates a sequence tha holds the queues, which are initialized the
 * first time they are fetched.
 *
 * The number of queues that exist in a Queue Pool is a certain number, and fetches
 * are obtaines in a sequential loop.
 * There are 32 of these queues per device, and when a queue is requested one of
 * these queues is returned round-robin. That is, the first queue requested is at
 * index 0, the second at index 1... to index 31, the index 0 again.
 *
 * There is a task maximum concurrency limit for MLU device computing tasks.
 * The maximum number of supports is the number of cores deivided by the minimum
 * split cell for the corresponding task.
 * The recommended maximum setting value is the number of corresponding device
 * clusters. for example, the maximum value of MLU270 is 4. Queue Pool can also
 * be used later for better devices.
 * When the number of computed queues is greater than the mainum support,
 * the computed tasks will be blocked. You can also get queue to do asynchronout
 * copy task or other tasks.
 *
 */
Queue getQueueFromPool(DeviceIndex device_index = -1);

/**
 *  Get the default MLU Queue.
 *  Get the MLU Queue specifying the corresponding device or current device,
 *      with the value -1 indicating the current device.
 *  Default Queue represents a queueu used if you do not specify a queue explicitly.
 *  Each hardware device has a default queue, which will be initialized the first
 *  time it goes back.
 */
Queue getDefaultQueue(DeviceIndex device_index = -1);

/*
 * Get the current MLU Queue.
 * Get the MLU Queue specifying the corresponding device or current device,
 *      with the value -1 indicating the current device.
 * Current Queue is initialized with the default queue, and if 'setCurrentQueue'
 * is invoked, the newly set Queue will be used as the current queue.
 */
Queue getCurrentQueue(DeviceIndex device_index = -1);

/**
 * The Queue passed in as the current queue of is corresponding device.
 */
void setCurrentQueue(Queue queue);

/**
 * Get the cnrtQueue_t corresponding to the current MLU Queue.
 * Get the MLU Queue specifying the corresponding device or current device,
 *      with the value -1 indicating the current device.
 *
 * This function can be used directly to get cnrtQueue_t, or it be used to
 * get the Queue object using 'getCurrentQueue' to get the corresponding cnrtQueue_t.
 * Interface usage is the same as 'getCurrentQueue().queue()'
 */
cnrtQueue_t getCurQueue(DeviceIndex device_index = -1);

/**
 * Function should be blocked until all precedent tasks in the queue are completed.
 */
inline void syncQueue(cnrtQueue_t queue) {
  TORCH_CNRT_CHECK(cnrtQueueSync(queue));
}

std::ostream& operator<<(std::ostream& stream, const Queue& queue);

/**
 * Represents an abstract object of the MLU Queue, which comes with some
 * encapsulation and additional functionalityfor cnrtQueue_t.
 * cnrtQueue_t lifecycle is created and destory by the internal abstract
 * class MLUQueueInternals.
 */
class Queue {
 public:
  Queue() {}
  explicit Queue(DeviceIndex device_index):
      device_index_(device_index), queue_id_(-1) {
         queue_ = getQueueFromPool(device_index).queue();
      }
  // The constructor encapsulates cnrtQueue_t and its corresponding property.
  Queue(cnrtQueue_t queue, DeviceIndex device_index, QueueIndex id):
    queue_(queue), device_index_(device_index), queue_id_(id) {}
  ~Queue() {}

  // Gets the attributes for Queue
  cnrtQueue_t queue() const { return queue_; }
  QueueIndex id() const { return queue_id_; }
  DeviceIndex device_index() const { return device_index_; }
  Device device() { return Device(DeviceType::MLU, device_index()); }

  // Add object instance comparison judgment.
  bool operator==(const Queue& other) const noexcept {
    return (this->device_index() == other.device_index()) &&
      (this->id() == other.id()) && (this->queue() == other.queue());
  }
  bool operator!=(const Queue& other) const noexcept {
    return !((this->device_index() == other.device_index()) &&
      (this->id() == other.id()) && (this->queue() == other.queue()));
  }

  // Function should be blocked until all precedent tasks in the queue are completed.
  void synchronize() {
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtQueueSync(queue_));
  }

  bool query() const {
    torch_mlu::mlu::MLUGuard guard(device_index_);
    cnrtRet_t err = cnrtQueryQueue(queue_);
    if (err == CNRT_RET_SUCCESS) {
        return true;
    }  else if (err != cnrtErrorBusy) {
        TORCH_CNRT_CHECK(err);
    }
    return false;
  }

  // The purpose of this function is to more conveniently permit binding
  // of Stream to and from Python.  Without packing, I have to setup a whole
  // class with two fields (device and queue id); with packing I can just
  // store a single uint64_t.
  //
  // The particular way we pack streams into a uint64_t is considered an
  // implementation detail and should not be relied upon.
  uint64_t pack() const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(QueueIndex) == 2, "QueueIndex is not 16-bit");
    static_assert(sizeof(DeviceIndex) == 2, "DeviceIndex is not 16-bit");
    static_assert(sizeof(DeviceType) == 4, "DeviceType is not 32-bit");
    // Concat these together into a 64-bit integer
    // See Note [Hazard when concatenating signed integers]
    uint64_t bits =
        static_cast<uint64_t>(static_cast<uint16_t>(id())) << 48
      | static_cast<uint64_t>(static_cast<uint16_t>(device_index())) << 32
      | static_cast<uint64_t>(static_cast<uint32_t>(DeviceType::MLU));
    return bits;
  }

 public:
  DeviceIndex device_index_;

 private:
  cnrtQueue_t queue_;

  // queue_id_ is the location annotation for the Queue, with a value of -1
  // for the default queue.
  QueueIndex queue_id_;
};
}  // namespace torch_mlu

namespace std {
template <>
struct hash<torch_mlu::Queue> {
  size_t operator()(torch_mlu::Queue s) const noexcept {
    return std::hash<uint64_t>{}(s.pack());
  }
};
}   // namespace std
