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
#include "aten/device/mlu_guard.h"
#include "aten/device/queue.h"
#include "cnrt.h"  // NOLINT

namespace torch_mlu {
struct Notifier {
  Notifier() {}
  ~Notifier() {
    if (is_created_) {
      destroyNotifier();
    }
  }
  Notifier(const Notifier&) = delete;
  Notifier& operator=(const Notifier&) = delete;
  Notifier(Notifier&& other) { moveHelper(std::move(other)); }
  Notifier& operator=(Notifier&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  void place(const Queue& queue);

  void place() { place(getCurrentQueue()); }

  void placeOnce(const Queue& queue);

  float elapsed_time(const Notifier& other) const;

  float hardware_time(const Notifier& other) const;

  void wait(const Queue& queue);

  bool query() const;

  void synchronize();

  bool isCreated() const { return is_created_; }

  c10::DeviceIndex device_index() const { return device_index_; }

  cnrtNotifier_t notifier() { return notifier_; }

 private:
  int device_index_ = -1;
  cnrtNotifier_t notifier_;
  void createNotifier(DeviceIndex device_index);
  void destroyNotifier();
  bool is_created_ = false;
  bool was_placed_ = false;

  void moveHelper(Notifier&& other);
};

}  // namespace torch_mlu
