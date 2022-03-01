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
#include <c10/util/Exception.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include "aten/util/memory_allocator.h"
#include "interface_builder.h"

namespace torch_mlu {
namespace jit {
namespace utils {

struct MemoryDeleter {
  void operator()(void* obj) {
    torch_mlu::memory::deallocateMemory<void>(obj);
  }
};
using memory_unique_ptr = std::unique_ptr<void, MemoryDeleter>; 

std::string getOpFullNameFromNode(const torch::jit::Node* node);

void saveTensor(const at::Tensor ival, const std::string& file);

at::Tensor loadTensor(const std::string& file);

void* accessTensor(const at::Tensor &tensor);

void setQuantizedParams(float scale, int qmode, int use_symmetry, std::vector<float>& params);

void setQuantizedParamsPerAxis(
        const std::vector<float>& scales, int qmode, int use_symmetry, std::vector<float>& params);

void getDynamicRange(
        const std::vector<float>& scales, int qmode, int use_symmetry, size_t bitwidth,
        std::vector<magicmind::Range>& input_range, std::vector<magicmind::Range>& weight_range);

} // namespace utils
} // namespace jit
} // namespace torch_mlu

#define MM_CHECK(status)                                  \
  do {                                                    \
    TORCH_CHECK((status) == magicmind::Status::OK(),      \
        "mm failure: ", status);          \
  } while (0)
