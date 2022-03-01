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

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "aten/util/exceptions.h"
#include "aten/device/device.h"
#include "aten/util/common.h"
#include "cnrt.h"  // NOLINT

namespace torch_mlu {
namespace mlu {

struct MLUGuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_type = at::DeviceType::MLU;
  MLUGuardImpl() {}
  explicit MLUGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == at::DeviceType::MLU);
  }
  at::DeviceType type() const override {
    return at::DeviceType::MLU;
  }

  c10::Device exchangeDevice(c10::Device device) const override {
    AT_ASSERT(device.type() == at::DeviceType::MLU);
    c10::Device old_device = getDevice();
    if (old_device.index() != device.index()) {
      setDevice(device);
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(at::DeviceType::MLU, current_device());
  }

  void setDevice(c10::Device device) const override {
    if ( GET_MLU_DEVICE < 0 ) return;
    AT_ASSERT(device.type() == at::DeviceType::MLU);
    TORCH_CNRT_CHECK(cnrtSetDevice(device.index()));
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    if ( GET_MLU_DEVICE < 0 ) return;
    TORCH_CNRT_WARN(cnrtSetDevice(device.index()));
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    LOG(ERROR) << "MLUGuardImpl::exchangeStream is not implemented";
    return s;
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }
};

}  // namespace mlu
}  // namespace torch_mlu
