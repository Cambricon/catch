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

#include <memory>
#include <mutex>

#include "aten/util/python_interface.h"

#include "cnrt.h"   // NOLINT
#include "cndev.h"  // NOLINT

#include "aten/util/exceptions.h"

namespace torch_mlu {

#define SINGLETON(CLASS)                            \
  public:                                           \
   CLASS(const CLASS&) = delete;                    \
   CLASS& operator=(const CLASS&) = delete;         \
   static CLASS& instance() {                       \
     static CLASS instance;                         \
     return instance;                               \
   }                                                \
  private:                                          \
   CLASS();                                         \
   ~CLASS()

#define GET_RUNNING_MODE                            \
  PythonInterface::instance().getRunningMode()

#define SET_RUNNING_MODE(MODE)                      \
  PythonInterface::instance().setRunningMode(MODE)

#define GET_CORE_NUMBER                             \
  Global::instance().getCoreNumber()

#define GET_CORE_VERSION                            \
  Global::instance().getCoreVersion()

#define GET_MLU_DEVICE                              \
  Global::instance().getDevice()

#define GET_INPUT_FORMAT                            \
  Global::instance().getInputFormat()

// Humanity will defeat COVID-19 after all!
// Running mode for MLU devicie.
// enum class RunningMode { CNML_EAGER, CNML_FUSE, CNNL };

// A singleton class to hold common Catch stuff
class Global {
 SINGLETON(Global);

 public:
  // Get MLU device index
  inline int getDevice() { return PythonInterface::getDevice(); }
  cndevNameEnum_t getDeviceName() {return device_name_;}
  bool isUsingFloatingDevice() {return is_running_fp32_;}
  void setFP32RunningMode(bool run_fp32) {is_running_fp32_ = run_fp32;}

 private:
  cndevNameEnum_t device_name_;
  bool is_running_fp32_;
};

enum class VIEWOPNAME {
  slice,
  permute,
  unfold,
  unsqueeze,
  squeeze,
  select,
  expand,
  reshape,
  view
};
}  // namespace torch_mlu
