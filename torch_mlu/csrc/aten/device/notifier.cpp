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

#include "aten/device/notifier.h"
#include "aten/device/queue.h"
#include "aten/util/python_interface.h"

namespace torch_mlu {

void Notifier::place(const Queue& queue) {
  if (!is_created_) {
    createNotifier(queue.device_index());
  }
  TORCH_CHECK(device_index_ == queue.device_index(), "Notifier device ",
              device_index_, " does not match placing queue's device ",
              queue.device_index(), ".");
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtPlaceNotifier(notifier_, queue.queue()));
  was_placed_ = true;
}

void Notifier::placeOnce(const Queue& queue) {
  if (!was_placed_) place(queue);
}

float Notifier::elapsed_time(const Notifier& other) const {
  TORCH_CHECK(is_created_ && other.isCreated(),
              "Both notifiers must be placed before calculating elapsed time.");
  float time_ms = 0;
  TORCH_CNRT_CHECK(cnrtNotifierElapsedTime(notifier_, other.notifier_, &time_ms));
  return time_ms;
}

float Notifier::hardware_time(const Notifier& other) const {
  TORCH_CHECK(is_created_ && other.isCreated(),
              "Both notifiers must be placed before calculating elapsed time.");
  float time_us = 0;
  TORCH_CNRT_CHECK(cnrtNotifierDuration(notifier_, other.notifier_, &time_us));
  return time_us;
}

void Notifier::synchronize() {
  if (is_created_) {
    TORCH_CNRT_CHECK(cnrtWaitNotifier(notifier_));
  }
}

// set MLUGuard before using this interface.
bool Notifier::query() const {
  if (!is_created_) {
    return true;
  }
  cnrtRet_t err = cnrtQueryNotifier(notifier_);
  if (err == CNRT_RET_SUCCESS) {
    return true;
  } else if (err != CNRT_RET_ERR_BUSY) {
    TORCH_CNRT_CHECK(err);
  }
  return false;
}

void Notifier::wait(const Queue& queue) {
  if (is_created_) {
    torch_mlu::mlu::MLUGuard guard(queue.device_index());
    TORCH_CNRT_CHECK(cnrtQueueWaitNotifier(notifier_, queue.queue(), 0));
  }
}

void Notifier::moveHelper(Notifier&& other) {
  std::swap(is_created_, other.is_created_);
  std::swap(was_placed_, other.was_placed_);
  std::swap(notifier_, other.notifier_);
  std::swap(device_index_, other.device_index_);
}

void Notifier::destroyNotifier() {
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtNotifierDestroy(notifier_));
}

void Notifier::createNotifier(DeviceIndex device_index) {
  device_index_ = device_index;
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtNotifierCreate(&notifier_));
  is_created_ = true;
}

}  // namespace torch_mlu
