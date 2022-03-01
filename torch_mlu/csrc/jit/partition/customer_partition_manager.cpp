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

#include "jit/partition/customer_partition_manager.h"
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <ATen/record_function.h>
#include "jit/partition/passes/segment_graph.h"
#include "jit/runtime/runner_cache.h"
#include "jit/interface.h"

namespace torch_mlu {
namespace jit {
namespace partition {

void RegisterCustomerPartitionPasses() {
  if (!torch_mlu::jit::canFuseOnMLU()) return;

  // place the passes of magicmind in this place.
  RegisterCustomOperators();
  torch::jit::registerPrePass(SegmentGraph);
}

void RegisterCustomOperators() {
  static torch::jit::RegisterOperators reg({
      // implement this MLUFusionGroup operator
      torch::jit::Operator(
          c10::Symbol::fromQualString("torch_mlu::MLUFusionGroup"),
          [](const torch::jit::Node* node) -> torch::jit::Operation {
            const auto key = torch_mlu::jit::runtime::RunnerCache::registerMLUFusion(node);
            return [key](torch::jit::Stack& stack) -> int {
              // input an empty stack to get the key outside the fuction
              if (stack.size() == 0) {
                // LOG_ERROR
                return 0;
              }
              auto maybe_spec = torch_mlu::jit::runtime::RunnerCache::retrieve(key);
              AT_ASSERT(maybe_spec);
              (*maybe_spec)->runFusion(stack);
              return 0;
            };
          },
          c10::AliasAnalysisKind::PURE_FUNCTION),
  });
}

}  // namespace partition
}  // namespace jit
}  // namespace torch_mlu
