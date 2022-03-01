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

#include "jit/runtime/runner_cache.h"

namespace torch_mlu {
namespace jit {
namespace runtime {

at::optional<MMGraphRunner*> RunnerCacheImpl::nolock_retrieve(int64_t key) {
  auto it = specMap_.find(key);
  if (it == specMap_.end()) {
    return at::nullopt;
  }
  return &it->second;
}

at::optional<MMGraphRunner*> RunnerCacheImpl::retrieve(int64_t key) {
  std::lock_guard<std::mutex> guard{mutex_};
  return nolock_retrieve(key);
}

// require the input graph is already normalized
at::optional<MMGraphRunner*> RunnerCacheImpl::retrieve(std::shared_ptr<torch::jit::Graph> graph) {
  std::string repr = graph->toString(false);
  std::lock_guard<std::mutex> guard{mutex_};
  auto it = graphToKey_.find(repr);
  if (it == graphToKey_.end())
    return at::nullopt;
  return nolock_retrieve(it->second);
}

int64_t RunnerCacheImpl::store(std::shared_ptr<torch::jit::Graph> graph) {
  std::string repr = graph->toString(false);
  std::lock_guard<std::mutex> guard{mutex_};
  const auto key = kernel_counter++;
  specMap_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
      std::forward_as_tuple(key, graph));
  graphToKey_.emplace(std::make_pair(std::move(repr), key));
  return key;
}

std::shared_ptr<torch::jit::Graph> RunnerCache::normalizeGraphForCache(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  auto result = torch::jit::Canonicalize(graph, /*keep_unique_names=*/false);
  torch::jit::EraseShapeInformation(result);
  return result;
}

int64_t RunnerCache::registerMLUFusion(const torch::jit::Node* fusion_group) {
  auto subgraph = fusion_group->g(torch::jit::attr::Subgraph);
  auto normalized_graph = normalizeGraphForCache(subgraph);
  const auto maybe_spec = Impl::Instance().retrieve(normalized_graph);

  if (maybe_spec) {
    return (*maybe_spec)->key();
  }
  return Impl::Instance().store(normalized_graph);
}

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu
