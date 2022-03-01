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

#include <c10/core/Device.h>
#include <string>
#include "aten/util/exceptions.h"
#include "aten/util/common.h"
#include "aten/util/version.h"
#include "cnrt.h" //NOLINT

#define MLU_DEVICE_NUM_MAX 16

namespace torch_mlu {

enum class DeviceType { CPU, GPU, MLU };

struct Device {
  Device() = default;
  explicit Device(const std::string& device_spec);
  Device(DeviceType hw_type, int ordinal)
      : hw_type(hw_type), ordinal(ordinal) {}

  DeviceType hw_type = DeviceType::CPU;
  int ordinal = 0;
};

const Device* GetDefaultDevice();

inline c10::DeviceIndex device_count() {
  // PythonInterface::initDevice();
  unsigned count;
  TORCH_CNRT_CHECK(cnrtGetDeviceCount(&count));
  return static_cast<c10::DeviceIndex>(count);
}

inline void setDevice(c10::DeviceIndex device_index) {
  if (GET_MLU_DEVICE < 0) return;
  TORCH_CNRT_CHECK(cnrtSetDevice(device_index));
  static std::once_flag flag;
  std::call_once(flag, checkRequirements);
}

c10::DeviceIndex current_device();

inline void synchronize() { TORCH_CNRT_CHECK(cnrtSyncDevice()); }

uint32_t getDeviceAttr(cnrtDeviceAttr_t attr);

struct DeviceProp{
    std::string name = "";
    int major = -1;
    int minor = -1;
    long total_memory = -1;
};

DeviceProp* getDeviceProperties(int64_t device);

}  // namespace torch_mlu
