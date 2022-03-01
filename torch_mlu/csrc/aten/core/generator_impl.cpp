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

#include <aten/core/generator_impl.h>
#include <aten/core/tensor_impl.h>
#include <aten/util/tensor_util.h>
#include <ATen/Utils.h>
#include "aten/device/device.h"

namespace torch_mlu {

// Ensures we only call mluGetDeviceCount only once
static std::once_flag num_mlu_init_flag;

// Total number of mlus in the system.
static int64_t num_mlus;

// Ensures default_gens_mlu is initialized once.
static std::deque<std::once_flag> mlu_gens_init_flag;

// Default, global MLU generators, one per MLU.
static std::vector<at::Generator> default_gens_mlu;

// Discriminate floating device type.
static bool is_floating_device = true;

/*
* Populates the global variables related to MLU generators
* Warning: this function must only be called once!
*/
static void initMLUGenVector() {
  num_mlus = device_count();
  mlu_gens_init_flag.resize(num_mlus);
  default_gens_mlu.resize(num_mlus);
  is_floating_device = Global::instance().isUsingFloatingDevice();
}

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultMLUGenerator gets the default generator for a particular
 * mlu device.
 */
const at::Generator& getDefaultMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_mlus);
  }
  std::call_once(mlu_gens_init_flag[idx], [&] {
    default_gens_mlu[idx] = at::make_generator<MLUGeneratorImpl>(idx);
    default_gens_mlu[idx].seed();
  });
  return default_gens_mlu[idx];
}

/**
 * Utility to create a MLUGeneratorImpl. Returns a shared_ptr
 */
at::Generator createMLUGenerator(at::DeviceIndex device_index) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  at::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_mlus, "The device_index is invalid.");
  auto generator = at::make_generator<MLUGeneratorImpl>(idx);
  auto gen_impl = at::check_generator<MLUGeneratorImpl>(generator);
  gen_impl->set_current_seed(c10::default_rng_seed_val);
  return generator;
}

/**
 * MLUGeneratorImpl class implementation
 */
MLUGeneratorImpl::MLUGeneratorImpl(at::DeviceIndex device_index)
  : c10::GeneratorImpl{at::Device(at::DeviceType::MLU, device_index),
      at::DispatchKeySet(c10::DispatchKey::MLU)}, state_need_reset_(true) {}

/**
 * Sets the seed to be used by MTGP
 *
 * See Note [Acquire lock when using random generators]
 */
void MLUGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  state_need_reset_ = true;
}

/**
 * Gets the current seed of MLUGeneratorImpl.
 */
uint64_t MLUGeneratorImpl::current_seed() const {
  return seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 */
uint64_t MLUGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/*
 * Gets the DeviceType of MLUGeneratorImpl.
 * Used for type checking during run time.
 */
at::DeviceType MLUGeneratorImpl::device_type() {
  return at::DeviceType::MLU;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<MLUGeneratorImpl> MLUGeneratorImpl::clone() const {
  return std::shared_ptr<MLUGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
MLUGeneratorImpl* MLUGeneratorImpl::clone_impl() const {
  auto gen = new MLUGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  auto state = this->state_;
  auto state_clone = state.clone();
  gen->set_state(state_clone);
  gen->set_state_flag(this->state_need_reset_);
  return gen;
}

/**
 * get_init_state_flag
 *
 * See Note [Acquire lock when using random generators]
  */

void MLUGeneratorImpl::delay_init_state_once() {
  // resize and set the state tensor.
  // TODO(zhangguopeng): the nullptr arg of cnnlRandGetMTGP32StateSize maybe not right.
  if (is_floating_device) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::call_once(init_state_flag, [&] {
      size_t state_size = 0;
      TORCH_CNNL_CHECK(cnnlRandGetMTGP32StateSize(nullptr, &state_size));
      auto options = at::TensorOptions().device(device_).dtype(at::kByte);
      state_ = at::empty(state_size, options);
    });
  }
}

/**
 * update_state
 *
 * See Note [Acquire lock when using random generators]
  */

void MLUGeneratorImpl::update_state() {
  // update the state tensor.
  if (is_floating_device && state_need_reset_) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto state_impl = getMluTensorImpl(state_);
    auto state_ptr = state_impl->cnnlMalloc();
    TORCH_CHECK(state_ptr, "the state point is nullptr, "
                           "please init state before calling its point");
    auto handle = getCurrentHandle();
    TORCH_CNNL_CHECK(cnnlRandMakeMTGP32KernelState(handle, state_ptr, nullptr, nullptr, seed_));
    state_need_reset_ = false;
  }
}

/**
 * get_state_ptr
 *
 * See Note [Acquire lock when using random generators]
  */

void* MLUGeneratorImpl::get_state_ptr() const {
  // update the state tensor.
  auto state_impl = getMluTensorImpl(state_);
  auto state_ptr = state_impl->cnnlMalloc();
  TORCH_CHECK(state_ptr, "the state point is nullptr, please init state before calling its point");
  return state_ptr;
}

/**
 * get state
 *
 * See Note [Acquire lock when using random generators]
  */
at::Tensor MLUGeneratorImpl::get_state() {
  delay_init_state_once();
  update_state();
  return state_;
}

/**
 * set state
 *
 * See Note [Acquire lock when using random generators]
  */
void MLUGeneratorImpl::set_state(at::Tensor state) {
  state_ = state;
  state_need_reset_ = false;
}

/**
 * set state flag
 *
 * See Note [Acquire lock when using random generators]
  */
void MLUGeneratorImpl::set_state_flag(bool flag) {
  state_need_reset_ = flag;
}

/**
 * set manual seed
 *
 * See Note [Acquire lock when using random generators]
  */
void manual_seed(uint64_t seed) {
  auto defaultGenerator = getDefaultMLUGenerator();
  std::lock_guard<std::mutex> lock(defaultGenerator.mutex());
  defaultGenerator.set_current_seed(seed);
}

void manual_seed_all(uint64_t seed) {
  std::call_once(num_mlu_init_flag, initMLUGenVector);
  for (int i = 0; i < num_mlus; i++) {
    auto mlu_gen = getDefaultMLUGenerator(at::DeviceIndex(i));
    mlu_gen.set_current_seed(seed);
  }
}

}  // namespace torch_mlu
