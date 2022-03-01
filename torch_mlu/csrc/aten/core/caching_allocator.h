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
#include "aten/device/queue.h"
#include "aten/device/caching_notifier.h"

namespace torch_mlu {

struct MLUCachingAllocator : public c10::Allocator {
  c10::DataPtr allocate(size_t size) const override;
  c10::DataPtr allocate(size_t size, c10::DeviceIndex device_id) const;
  c10::DeleterFnPtr raw_deleter() const override;
};

struct BoundException : public std::exception {
  const char * what() const throw() {
    return "MLU memory out of bounds!";
  }
};

struct ManageException : public std::exception {
  const char * what() const throw() {
    return "MLU memory out of allocator!";
  }
};

c10::Allocator* getMLUCachingAllocator(void);
uint64_t currentMemoryAllocated(int device_id);
uint64_t currentMemoryCached(int device_id);
uint64_t maxMemoryAllocated(int device_id);
uint64_t maxMemoryCached(int device_id);
void emptyCachedMem();
void setDebugEnv(char* flag);
void memoryDebug(c10::DataPtr* data);
void memoryDebug(const c10::DataPtr* data);
void memoryDebug();
void recordQueue(const c10::DataPtr& ptr, Queue queue);
bool get_memory_strategy();
void set_memory_strategy(bool ms);

}  // namespace torch_mlu
