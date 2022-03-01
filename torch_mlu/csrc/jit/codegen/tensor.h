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

#include <torch/custom_class.h>
#include "interface_builder.h"  // NOLINT
#include "jit/codegen/codegen.h"
#include "jit/utils/data_type.h"

namespace torch_mlu {
namespace jit {
namespace codegen {

struct ITensorContainer : torch::CustomClassHolder {
  enum struct Type {
    Invalid,
    ITensor,
    ITensorVector,
  };
  ITensorContainer() : _type(Type::Invalid), _tensor(nullptr) {}

  bool isITensor() {
    return Type::ITensor == _type;
  }

  bool isITensorVector() {
    return Type::ITensorVector == _type;
  }

  void set_tensor(magicmind::ITensor* tensor) {
    _tensor = tensor;
    _type = Type::ITensor;
  }

  void set_tensor_vector(const std::vector<magicmind::ITensor*>& vec) {
    _tensor_vec = vec;
    _type = Type::ITensorVector;
  }

  magicmind::ITensor* tensor() {
    return _tensor;    
  }

  std::vector<magicmind::ITensor*> tensor_vector() {
    return _tensor_vec;
  }

  magicmind::ITensor* _tensor;
  std::vector<magicmind::ITensor*> _tensor_vec;
  Type _type;
};

bool isITensor(const torch::jit::IValue& ivalue);

magicmind::ITensor* getITensor(const torch::jit::IValue& ivalue);

std::vector<magicmind::ITensor*> getITensorVector(const torch::jit::IValue& ivalue);

magicmind::ITensor* getOrCreateITensor(MagicmindHandle *handle, const torch::jit::IValue& ivalue);

magicmind::ITensor* createConstITensor(MagicmindHandle *handle, at::Tensor tensor);

torch::jit::IValue bindITensor(magicmind::ITensor* tensor);

at::Tensor getEmptyTensor(magicmind::ITensor* tensor);

torch::jit::IValue bindITensorVector(const std::vector<magicmind::ITensor*>& tensor_vec);
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
