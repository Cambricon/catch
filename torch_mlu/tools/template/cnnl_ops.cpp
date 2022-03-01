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

#include "aten/operators/cnnl_ops.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/util/cndumper.h"
#include "aten/util/python_interface.h"
#include "cnnl_ops.h"  // NOLINT

const bool CATCH_EXCEPTION = std::getenv("ENABLE_CNNL_TRYCATCH") != nullptr &&
  (strcmp(std::getenv("ENABLE_CNNL_TRYCATCH"), "off") == 0 ||
  strcmp(std::getenv("ENABLE_CNNL_TRYCATCH"), "OFF") == 0 ||
  strcmp(std::getenv("ENABLE_CNNL_TRYCATCH"), "0") == 0)
                       ? false   // dispatch to CNNL Op data path
                       : true;  // dispatch to try-catch data path by default
#ifdef DEBUG
  #define MLU_KERNEL_DISPATCH(BACKEND, OP, ...)           \
  auto&& mlu_result = BACKEND::ops::OP(__VA_ARGS__);
#else
  #define MLU_KERNEL_DISPATCH(BACKEND, OP, ...)           \
  auto&& mlu_result = BACKEND::ops::OP(__VA_ARGS__);      \
  if (!PythonInterface::getAsyncMode()) {                 \
      auto queue = getCurrentQueue();                     \
      queue.synchronize();                                \
  }
#endif

#define CPU_KERNEL_DISPATCH(CPU_OP, ...)                  \
  auto&& cpu_result = OpMethods::CPU_OP(__VA_ARGS__);

#define SWITCH_TO_CPU                                   \
  do {                                                  \
    auto envar = std::getenv("TEST_CPU_DISPATCH");      \
    if (envar != nullptr &&                             \
        (strcmp(envar, "1") == 0 ||                     \
        strcmp(envar, "on") == 0 ||                     \
        strcmp(envar, "ON") == 0)) {                    \
      throw std::exception();                           \
    }                                                   \
  } while (0);


#define MLU_DISPATCH(BACKEND, OP, MLU_OP, ...)           \
if USE_CPU_COMPARE(OP) {                                 \
    MLU_KERNEL_DISPATCH(BACKEND, MLU_OP, __VA_ARGS__)    \
    DUMP_RESULT(#BACKEND, OP, mlu_result)                \
  if (NOT_INPLACE(#OP)) {                                \
    DUMP_TRY(CPU_KERNEL_DISPATCH(OP, __VA_ARGS__)        \
             DUMP_RESULT("cpu", OP, cpu_result) )        \
  }                                                      \
  DUMP_FINISH(OP)                                        \
  return mlu_result;                                     \
} else if (CATCH_EXCEPTION) {                            \
  try {                                                  \
    if (strcmp(#MLU_OP, "cnnl_copy_") != 0) {            \
      SWITCH_TO_CPU                                      \
    }                                                    \
    MLU_KERNEL_DISPATCH(BACKEND, MLU_OP, __VA_ARGS__)    \
    DUMP_RESULT(#BACKEND, OP, mlu_result)                \
    DUMP_FINISH(OP)                                      \
    return mlu_result;                                   \
  } catch (c10::Error & e) {                             \
    std::cerr << e.what_without_backtrace() << std::endl;\
    CPU_KERNEL_DISPATCH(OP, __VA_ARGS__)                 \
    DUMP_RESULT("cpu", OP, cpu_result)                   \
    DUMP_FINISH(OP)                                      \
    return cpu_result;                                   \
  } catch (std::exception & e) {                         \
    CPU_KERNEL_DISPATCH(OP, __VA_ARGS__)                 \
    DUMP_RESULT("cpu", OP, cpu_result)                   \
    DUMP_FINISH(OP)                                      \
    return cpu_result;                                   \
  }                                                      \
} else {                                                 \
  MLU_KERNEL_DISPATCH(BACKEND, MLU_OP, __VA_ARGS__)      \
  DUMP_RESULT(#BACKEND, OP, mlu_result)                  \
  DUMP_FINISH(OP)                                        \
  return mlu_result;                                     \
}

#define CNNL_DISPATCH(OP, CNNL_OP, ...)                  \
  MLU_DISPATCH(cnnl, OP, CNNL_OP, __VA_ARGS__)

#define BANG_DISPATCH(OP, BANG_OP, ...)                  \
  MLU_DISPATCH(bang, OP, BANG_OP, __VA_ARGS__)

namespace torch_mlu {
${cnnl_ops_definitions}  // NOLINT

}  // namespace torch_mlu
