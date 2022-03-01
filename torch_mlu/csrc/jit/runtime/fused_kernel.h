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

#include <torch/torch.h>
#include "jit/codegen/codegen.h"
#include "interface_runtime.h"  // NOLINT

namespace torch_mlu {
namespace jit {
namespace runtime {

// mm do not care about input tensor memory format
struct MMTensorDesc {
  at::ScalarType scalar_type;
  std::vector<int64_t> sizes_;

  // Delegating constructors
  MMTensorDesc(
      const at::ScalarType& type,
      const at::IntArrayRef& sizes)
      : scalar_type{type}, sizes_(sizes.begin(), sizes.end()) {}

  explicit MMTensorDesc(const at::Tensor& t)
      : MMTensorDesc(t.scalar_type(), t.sizes()) {}

  explicit MMTensorDesc(const c10::TensorTypePtr& type)
      : MMTensorDesc(
            type->scalarType().value(),
            type->sizes().concrete_sizes().value()) {}

  bool operator==(const MMTensorDesc& desc) const {
    return scalar_type == desc.scalar_type && sizes_ == desc.sizes_;
  }

  bool operator!=(const MMTensorDesc& desc) const {
    return !(*this == desc);
  }

  static size_t hash(const MMTensorDesc& spec) {
    return torch::get_hash(
        spec.scalar_type,
        torch::hash<std::vector<int64_t>>{}(spec.sizes_));
  }
};


struct MMArgSpec {
  MMArgSpec(at::TensorList inputs, int _device)
      : descs_{c10::fmap<MMTensorDesc>(inputs)},
        device_{_device},
        hash_code_{torch::get_hash(_device, inputs.size(), descs_)} {}

  // (Common) hash function
  static size_t hash(const MMArgSpec& spec) {
    return spec.hash_code_;
  }

  // Comparators
  bool operator==(const MMArgSpec& other) const {
    return (descs_ == other.descs_);
  }

  bool operator!=(const MMArgSpec& spec) const {
    return !(*this == spec);
  }

  // Getters
  size_t hashCode() const {
    return hash_code_;
  }
  const std::vector<MMTensorDesc>& descs() const {
    return descs_;
  }
  int device() const {
    return device_;
  }

 private:
  std::vector<MMTensorDesc> descs_;
  int device_;
  size_t hash_code_;
};

inline std::ostream& operator<<(std::ostream& out, const MMTensorDesc& d) {
  out << d.scalar_type << "[";
  for (const auto b : d.sizes_)
    out << b << ",";
  out << "]";
  return out;
}

struct MMFusedKernel {
  MMFusedKernel(codegen::magicmind_unique_ptr<magicmind::IEngine>&& ptr,
                codegen::magicmind_unique_ptr<magicmind::IContext>&& icontext,
                const std::vector<MMTensorDesc>& desc) :
      iengine_(std::move(ptr)), icontext_(std::move(icontext)), input_desc_(desc) {}
  std::vector<at::Tensor> launch_raw(int device_id, const std::vector<at::Tensor>& inputs);

 private:
  codegen::magicmind_unique_ptr<magicmind::IEngine> iengine_;
  codegen::magicmind_unique_ptr<magicmind::IContext> icontext_;
  const std::vector<MMTensorDesc> input_desc_;
};

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu
