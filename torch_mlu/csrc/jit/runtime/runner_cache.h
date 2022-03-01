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

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include "jit/runtime/graph_runner.h"

namespace torch_mlu {
namespace jit {
namespace runtime {

template <typename T>
class SingletonHolder {
  public:
  SingletonHolder(const SingletonHolder&) = delete;
  SingletonHolder& operator=(const SingletonHolder&) = delete;
  template<typename ... ArgT>
  static T& Instance(ArgT... args) {
    static T instance_{args...};
    return instance_;
  }
  private:
  SingletonHolder() = default;
  virtual ~SingletonHolder() {}
};

class RunnerCacheImpl {
  friend class SingletonHolder<RunnerCacheImpl>;
  at::optional<MMGraphRunner*> nolock_retrieve(int64_t key);
 public:
  at::optional<MMGraphRunner*> retrieve(std::shared_ptr<torch::jit::Graph> key);
  at::optional<MMGraphRunner*> retrieve(int64_t key);

  int64_t store(std::shared_ptr<torch::jit::Graph> graph);
 private:
  RunnerCacheImpl() = default;
  ~RunnerCacheImpl() = default;
  RunnerCacheImpl(const RunnerCacheImpl&) = delete;
  RunnerCacheImpl& operator=(const RunnerCacheImpl&) = delete;
  int64_t kernel_counter{0};
  // Map of fusion key to MMGraphRunner
  std::unordered_map<int64_t, MMGraphRunner> specMap_;

  // Map of pretty-printed graph string to fusion key
  // Used to check if a graph has already been cached in specMap_
  std::unordered_map<std::string, int64_t> graphToKey_;
  std::mutex mutex_;
};

/// this class is responsible for maintaining a
/// mapping from a given PyTorch JIT graph(canonicalized and erased shape information)
/// to MMGraphRunner used to compile and run graph in MagicMind.
class RunnerCache {
  using Impl = SingletonHolder<RunnerCacheImpl>;
public:
  // Canonicalize a graph, renumbering it so that all structurally equivalent
  // graphs have same numbers.
  static std::shared_ptr<torch::jit::Graph> normalizeGraphForCache(
    const std::shared_ptr<torch::jit::Graph>& graph);
  static at::optional<MMGraphRunner*> retrieve(int64_t key) {
    return Impl::Instance().retrieve(key);
  }
  static int64_t registerMLUFusion(const torch::jit::Node* fusion_group);
};

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu
