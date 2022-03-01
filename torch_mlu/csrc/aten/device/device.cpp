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

#include "aten/device/device.h"
#include "aten/device/mlu_guard.h"
#include "aten/util/python_interface.h"

namespace torch_mlu {
namespace {

constexpr uint32_t rem_for_stack = 128 * 1024;

std::string DeviceTypeToString(DeviceType hw_type) {
  switch (hw_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::GPU:
      return "GPU";
    case DeviceType::MLU:
      return "MLU";
  }
  LOG(ERROR) << "Invalid device type";
}

void ParseDevice(const std::string& device_spec, Device* device) {
  device->ordinal = 0;
  device->hw_type = DeviceType::MLU;
}

}  // namespace

Device::Device(const std::string& device_spec) {
  ParseDevice(device_spec, this);
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
  return default_device;
}

c10::DeviceIndex current_device() {
    int dev_ordinal;
    TORCH_CNRT_CHECK(cnrtGetDevice(&dev_ordinal));
    return static_cast<c10::DeviceIndex>(dev_ordinal);
}

uint32_t getDeviceAttr(cnrtDeviceAttr_t attr) {
  int dev_ordinal = 0;
  int device_attr = 1;
  TORCH_CNRT_CHECK(cnrtGetDevice(&dev_ordinal));
  TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_attr, attr, dev_ordinal));
  if (attr == cnrtAttrNramSizePerMcore) {
    device_attr -= rem_for_stack;
  }
  return device_attr;
}

at::DeviceIndex num_mlus = -1;
std::once_flag context_init_flag;
std::deque<std::once_flag> device_flags;
std::vector<DeviceProp> device_properties;

void initMLUContextVectors() {
    num_mlus = device_count();
    device_flags.resize(num_mlus);
    device_properties.resize(num_mlus);
}

void initDeviceProperty(at::DeviceIndex device_index) {
    torch_mlu::mlu::MLUGuard guard(device_index);
    
    // device attribute
    TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_properties[device_index].major,
                cnrtAttrComputeCapabilityMajor, device_index));
    TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(&device_properties[device_index].minor,
                cnrtAttrComputeCapabilityMinor, device_index));

    // memory info
    size_t available, total;
    TORCH_CNRT_CHECK(cnrtMemGetInfo(&available, &total));
    device_properties[device_index].total_memory = static_cast<long>(total);
    
    // device name
    cnrtDeviceInfo_t info;
    TORCH_CNRT_CHECK(cnrtGetDeviceInfo(&info, device_index));
    device_properties[device_index].name = info.device_name;
}

DeviceProp* getDeviceProperties(int64_t device) {
    std::call_once(context_init_flag, initMLUContextVectors);
    if (device == -1) device = current_device();
    AT_ASSERT(device >= 0 && device < num_mlus);
    std::call_once(device_flags[device], initDeviceProperty, device);
    return &device_properties[device];
}

}  // namespace torch_mlu
