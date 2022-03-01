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
#include "jit/runtime/fused_kernel.h"
#include "jit/utils/utils.h"

namespace torch_mlu {
namespace jit {
namespace runtime {

/// For a given PyTorch JIT graph(canonicalized and erased shape information),
/// this class is responsible for maintaining a mapping from
/// PyTorch input information to MMFusedKernel(contains compiled IModel) used to run that
/// graph in Magicmind.
class MMGraphRunner {
 public:
  explicit MMGraphRunner(const int64_t _key, const std::shared_ptr<torch::jit::Graph>& _graph)
      : key_{_key},
        graph_{_graph},
        code_{_graph, "<fused code>"} {}
  void runFusion(torch::jit::Stack& stack);
  bool runMLUFusion(torch::jit::Stack& stack);
  void runFallback(torch::jit::Stack& stack);
  const torch::jit::Code& code() const {
    return code_;
  }
  std::shared_ptr<torch::jit::Graph> graph() const {
    return graph_;
  }
  int64_t key() const {
    return key_;
  }

 private:
  enum class DebugType {
    CPU,
    CNNL,
    NONE
  };
  DebugType get_debug_type();
  int debug_fused_op(torch::jit::Stack& stack);
  std::vector<at::Tensor> launch_fusion(
      std::shared_ptr<MMFusedKernel> fusion,
      const std::vector<at::Tensor>& inputs,
      int device_id);
  std::shared_ptr<MMFusedKernel> compile_kernel(
      const MMArgSpec& arg_spec,
      const at::ArrayRef<torch::jit::IValue>& all_inputs,
      int dev_id);

  c10::optional<std::shared_ptr<MMFusedKernel>> find_kernel(
      const MMArgSpec& arg_spec) const {
    std::lock_guard<std::mutex> guard{mutex_};
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end())
      return c10::nullopt;
    return it->second;
  }

  void cache_kernel(const MMArgSpec& arg_spec, std::shared_ptr<MMFusedKernel> kernel)
      const {
    std::lock_guard<std::mutex> guard{mutex_};
    kernels_.emplace(arg_spec, kernel);
  }

  std::shared_ptr<torch::jit::Graph> graph_;
  int64_t key_;
  torch::jit::Code code_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<MMArgSpec, std::shared_ptr<MMFusedKernel>, torch::hash<MMArgSpec>>
      kernels_;
};

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu
