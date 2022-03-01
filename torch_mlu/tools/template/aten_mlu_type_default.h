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
#include <ATen/Tensor.h>
#include <ATen/core/function.h>

namespace torch_mlu {

const int MLU2CPU = 0;
const int CPU2MLU = 1;

// only cpu tensors are registered with the InplaceOpArgsRegistry!
class InplaceOpArgsRegistry {
private:
  std::vector<c10::IValue>& originalArgs_;

  struct TensorHash {
      // we don't bother to calculate the hashes for Tensors, because
      //  1. we can't not aquire the private Tensor::impl_ in a safe way
      //  2. the number of the tensors that need to be registered is very limited
      inline std::size_t operator()(const at::Tensor&) const noexcept {
          return 1;
      }
  };

  struct TensorComparator {
      bool operator()(const at::Tensor& tensor1, const at::Tensor& tensor2) const {
          return tensor1.is_same(tensor2);
      }
  };

  // for registration of Tensors,
  // positions are expressed in 'int' and are with respect to the original args[]
  std::unordered_map<at::Tensor, int, TensorHash, TensorComparator> uninary_map{};
  // for registration of Tensors in TensorLists
  // positions are expressed in 'std::pair<int, int>' and
  // the first index is with respect to the original args[]
  // the socond index is with respect to the TensorList at args[the first index]
  std::unordered_map<at::Tensor, std::pair<int, int>, TensorHash, TensorComparator> binary_map{};

  std::unordered_map<at::Tensor, bool, TensorHash, TensorComparator> mutable_map{};



public:
  explicit InplaceOpArgsRegistry(auto& args) : originalArgs_(args) {}

  void registerTensor(const at::Tensor& tensor, int index, bool isMutable) {
    // no move!
    // the registry shares the ownership of the tensors. otherwise is_same()
    // would be ineffective,
    // because when ownership transfers to unboxed-calls, the registered
    // tensor would become invalidated.
    uninary_map[tensor] = index;
    mutable_map[tensor] = isMutable;
  }

  void registerTensor(const at::Tensor& tensor, int index1, int index2, bool isMutable) {
    // no move!
    // the registry shares the ownership of the tensors. otherwise is_same()
    // would be ineffective,
    // because when ownership transfers to unboxed-calls, the registered tensor
    // would become invalidated.
    binary_map[tensor] = std::make_pair(index1, index2);
    mutable_map[tensor] = isMutable;
  }
  // input -- cpu tensor
  // output -- optional mlu tensor
  c10::optional<std::pair<at::Tensor, bool>> getOriginalMLUTensor(at::Tensor& tensor) {
    // find the reigstered tensor
    auto it1 = uninary_map.find(tensor);
    if (it1 != uninary_map.end())
      return std::make_pair(originalArgs_[it1->second].toTensor(), mutable_map[tensor]);

    // find the registered tensor in tensorLists
    auto it2 = binary_map.find(tensor);
    if (it2 != binary_map.end()) {
      std::pair<int, int> indexPair = it2->second;
      const c10::List<at::Tensor>& tensorList = originalArgs_[indexPair.first].toTensorList();
      return std::make_pair(tensorList.get(indexPair.second), mutable_map[tensor]);
    }
    return c10::nullopt;
  }
};

void RegisterAtenOperators();

void inputConvertAndPushTensor(torch::jit::Stack& stack, at::Tensor&& tensor, int direction,
                              InplaceOpArgsRegistry& reg, int index, bool isMutable);

void inputConvertAndPushTensor(c10::List<at::Tensor>& tensorList, at::Tensor&& tensor,
                int direction, InplaceOpArgsRegistry& reg, int index, bool isMutable);

void outputConvertAndPushTensor(torch::jit::Stack& stack, at::Tensor&& tensor, int direction,
                              InplaceOpArgsRegistry& reg);

void outputConvertAndPushTensor(c10::List<at::Tensor>& tensorList, at::Tensor&& tensor,
                              int direction, InplaceOpArgsRegistry& reg);
}  // namespace torch_mlu
