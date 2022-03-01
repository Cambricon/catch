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

#include "interface_builder.h"  // NOLINT

#include <vector>
#include <memory>
#include <unordered_map>
#include <torch/csrc/jit/ir/ir.h>
#include "jit/utils/utils.h"


namespace torch_mlu {
namespace jit {
namespace codegen {

struct MMDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->Destroy();
    }
  }
};
template <typename T>
using magicmind_unique_ptr = std::unique_ptr<T, MMDeleter>;
using utils::memory_unique_ptr;

class MagicmindHandle final {
 public:
  MagicmindHandle();

  ~MagicmindHandle();

  void bindingValueAndIvalue(const torch::jit::Value* value, torch::jit::IValue ivalue);

  magicmind_unique_ptr<magicmind::INetwork> network;
  magicmind_unique_ptr<magicmind::IBuilder> builder;
  magicmind_unique_ptr<magicmind::IBuilderConfig> builder_config;

  std::vector<void*> persistent_buffers;

  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> conversion_value_map;

  // When the input tensors of this graph have fp16 data type, has_fp16_input will be set.
  bool has_fp16_input = false;

  // Indicate the model will be inferenced on: Float(32)/Qint16(16)/Qint8(8), default is Float.
  int quantized_bitwidth = 32;

  // The MLU device id
  int device_id = 0;
};

magicmind_unique_ptr<magicmind::IModel> convertSubGraphToIModel(
    MagicmindHandle *handle,
    const torch::jit::Block* block,
    const at::ArrayRef<torch::jit::IValue>& stack);

}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
