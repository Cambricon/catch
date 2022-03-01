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

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include "aten/device/device.h"
#include "aten/cnnl/cnnlHandle.h"

namespace torch_mlu {
struct MLUGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  explicit MLUGeneratorImpl(at::DeviceIndex device_index = -1);
  ~MLUGeneratorImpl() = default;

  // MLUGeneratorImpl methods
  std::shared_ptr<MLUGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  static at::DeviceType device_type();
  void delay_init_state_once();
  at::Tensor get_state();
  void set_state(at::Tensor state);
  void set_state_flag(bool flag);
  void update_state();
  void* get_state_ptr() const;

private:
  MLUGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = c10::default_rng_seed_val;
  std::once_flag init_state_flag;
  at::Tensor state_;
  bool state_need_reset_;
};

const at::Generator& getDefaultMLUGenerator(at::DeviceIndex device_index = -1);
at::Generator createMLUGenerator(at::DeviceIndex device_index = -1);
void manual_seed(uint64_t seed);
void manual_seed_all(uint64_t seed);
}  // namespace torch_mlu
