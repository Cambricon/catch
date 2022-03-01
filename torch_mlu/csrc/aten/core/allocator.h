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

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <caffe2/core/logging.h>
#include "aten/util/exceptions.h"

static std::mutex mlu_mutex;

namespace torch_mlu {

static void MLUDeleter(void* ptr) {
  if (ptr) {
    TORCH_CNRT_CHECK(cnrtFree(ptr));
    ptr = nullptr;
  }
}

struct MLUAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t size) const {
    void* ptr = nullptr;
    return {ptr, ptr, &MLUDeleter, c10::DeviceType::MLU};
  }

  c10::DataPtr allocate(size_t size, c10::Device device) const {
    void* ptr = nullptr;
    return {ptr, ptr, &MLUDeleter, device};
  }

  c10::DataPtr allocate(size_t nbytes, int16_t device_index) const {
    std::lock_guard<std::mutex> lock(mlu_mutex);
    void* data = nullptr;
    TORCH_CNRT_CHECK(cnrtMalloc(&data, nbytes));
    return {data, data, &MLUDeleter, c10::Device(c10::DeviceType::MLU, device_index)};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &MLUDeleter;
  }
};

MLUAllocator* getMLUDeviceAllocator(void);

}  // namespace torch_mlu

namespace c10 {

struct DefaultMLUAllocator final : public c10::Allocator {
  DefaultMLUAllocator() {}
  ~DefaultMLUAllocator() override {
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    CAFFE_THROW("MLU dose not support malloc with nbytes.");
    return c10::DataPtr();
  }

  private:
  static void Delete(void* ptr) {
    if (ptr) {
      cnrtRet_t status = cnrtFree(ptr);
      if (status != CNRT_RET_SUCCESS) {
        LOG(FATAL) << "Error at: " << __FILE__ <<
          ":" << __LINE__ << ": MLUFree failed!";
      }
      ptr = nullptr;
    }
  }
};

}  // namespace c10
