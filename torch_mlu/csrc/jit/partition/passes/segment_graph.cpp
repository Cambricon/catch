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

#include "jit/partition/passes/segment_graph.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include "jit/interface.h"
#include "jit/codegen/convertion/convert.h"
#include "jit/codegen/evalution/eval.h"
#include "jit/utils/utils.h"
#include "aten/util/cnlog.h"

namespace torch_mlu {
namespace jit {
namespace partition {

// op_name is in format like aten::add.Tensor
std::vector<std::string> debugForcedFallbackOps() {
  const char* oplist = getenv("DEBUG_FORCED_FALLBACK_OPS");

  std::vector<std::string> black_list;
  if (!oplist) return black_list;
  std::string ops(oplist);
  std::istringstream iss(ops);
  std::string op;

  while (std::getline(iss, op, ',')) {
    black_list.emplace_back(std::move(op));
  }

  return black_list;
}

bool isSupported(const torch::jit::Node *ptNode) {
  if (!codegen::evalution::isEvalNode(ptNode) &&
      !codegen::convertion::isConvertNode(ptNode) &&
      ptNode->kind() != at::Symbol::fromQualString("torch_mlu::MLUFusionGroup")) {
    auto *maybe_schema = ptNode->maybeSchema();
    if (!maybe_schema) {
      CNLOG(DBG) << "Not supported for magicmind: " << ptNode->kind().toQualString();
    } else {
      CNLOG(DBG) << "Not supported for magicmind: " << *maybe_schema;
    }
    return false;
  }

  auto op_name = utils::getOpFullNameFromNode(ptNode);
  auto black_list = debugForcedFallbackOps();
  auto it = std::find(black_list.begin(), black_list.end(), op_name);
  if (it != black_list.end()) {
    return false;
  }

  return true;
}


torch::jit::value_list
sortReverseTopological(at::ArrayRef<torch::jit::Value *> inputs,
                       torch::jit::Block *block) {
  torch::jit::value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }

  std::sort(result.begin(), result.end(),
            [&](torch::jit::Value *a, torch::jit::Value *b) {
              return a->node()->isAfter(b->node());
            });
  return result;
}

bool canMerge(torch::jit::Node *node, FusionCallback fn,
              torch::jit::Node *consumer) {
  if (node->kind() == torch::jit::prim::Param) return false;
  // Check that the node is supported
  if (!fn(node)) {
    return false;
  }

  // If the node is a producer (has a consumer), check that all non-tensor
  // outputs are only consumed by the consumer.
  // If consumer is nullptr, check all outputs of node must be subtype of TesorType
  for (torch::jit::Value *output : node->outputs()) {
    if (output->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      continue;
    }
    if (!consumer) {
      return false;
    }
    // Producers can have non-tensor outputs as long as they are only consumed
    // by consumer. Consumers cannot have non-tensor outputs.
    for (auto use : output->uses()) {
      if (use.user != consumer) {
        return false;
      }
    }
  }
  return true;
}

bool canMergeProducer(torch::jit::Node *node, FusionCallback fn,
                  torch::jit::Node *consumer) {
  if (!fn(node) && node->kind() != torch::jit::prim::Constant &&
      node->kind() != torch::jit::prim::GetAttr) {
    return false;
  }
  return true;
}

bool canMergeConsumer(torch::jit::Node *consumer, FusionCallback fn) {
  if (consumer->kind() == torch::jit::prim::GetAttr ||
      consumer->kind() == torch::jit::prim::Constant ||
      consumer->kind() == torch::jit::prim::Param) return false;
  if (!fn(consumer)) return false;
  return true;
}

torch::jit::Node *tryFuse(torch::jit::Node *consumer,
                           torch::jit::Value *producer_value,
                           torch::jit::AliasDb &aliasDb, FusionCallback fn,
                           at::Symbol kind) {
  // if consumer is not supported, return
  if (!canMergeConsumer(consumer, fn)) {
    return nullptr;
  }
  auto producer = producer_value->node();
  // Check that producer can be merged
  // if producer is not supported and consumer is fusedkernel, return
  // if producer is not supported and consumer is supported(not fusedkernel),
  // go on to create single fusedkernel with consumer
  if (!canMerge(producer, fn, consumer) &&
     consumer->kind() == at::Symbol::fromQualString("torch_mlu::MLUFusionGroup")) return nullptr;

  // if consumer is not fusedkernel and has non-tensor output, return
  if (!(consumer->kind() == kind ||
      canMerge(consumer, fn, /*consumer*/ nullptr))) {
    return nullptr;
  }
  // Rearrange nodes such that all uses of producer are after the
  // consumer. Fusion will rewrite those later uses to use the version of
  // producer generated by the fused blob. In this case, producer becomes
  // an output of the fusion group.
  if (!aliasDb.moveBeforeTopologicallyValid(producer, consumer)) {
    return nullptr;
  }

  // if consumer has no subgraph and is not fusedkernel, create a fusedkernel contains consumer
  if (!consumer->hasAttribute(torch::jit::attr::Subgraph) &&
      consumer->kind() != kind) {
    consumer =
        torch::jit::SubgraphUtils::createSingletonSubgraph(consumer, kind);
  }
  // handle only single node is merged in fusedkernel
  // if producer is not supported, return
  if (producer->kind() == torch::jit::prim::Param || !fn(producer)) {
    return consumer;
  }

  if (producer->kind() == torch::jit::prim::Constant) {
    auto &subgraph = consumer->g(torch::jit::attr::Subgraph);
    torch::jit::Node *inConst = subgraph->createClone(
        producer, [](torch::jit::Value *) -> torch::jit::Value * {
          throw std::runtime_error("unexpected input to Constant node");
        });
    subgraph->insertNode(inConst);
  } else {
    torch::jit::SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}

std::pair<torch::jit::graph_node_list::iterator, bool>
scanNode(torch::jit::Node *consumer, torch::jit::AliasDb &aliasDb,
         torch::jit::Block *block, FusionCallback fn, at::Symbol kind) {
  if (fn(consumer)) {
    auto inputs = sortReverseTopological(consumer->inputs(), block);
    for (auto producer : inputs) {
      auto fusion_group = tryFuse(consumer, producer, aliasDb, fn, kind);
      if (fusion_group) {
        return std::make_pair(fusion_group->reverseIterator(), true);
      }
    }
  }
  return std::make_pair(++consumer->reverseIterator(), false);
}


void Partitioner::fuseNodesForMM(std::shared_ptr<torch::jit::Graph> graph) {
  c10::Symbol fuse_symbol =
    at::Symbol::fromQualString("torch_mlu::MLUFusionGroup");
  torch::jit::AliasDb aliasDb(graph);
  auto block = graph->block();

  bool any_changed;
  do {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool changed;
      std::tie(it, changed) =
          scanNode(*it, aliasDb, block, is_supported_, fuse_symbol);
      any_changed |= changed;
    }
  } while (any_changed);
}

void SegmentGraph(std::shared_ptr<torch::jit::Graph>& graph) {
  if (!canFuseOnMLU()) {return;}
  Partitioner partitioner(isSupported);
  partitioner.fuseNodesForMM(graph);
  torch::jit::EliminateDeadCode(graph);
  CNLOG(DBG) << *graph;
  return;
}

}  // namespace partition
}  // namespace jit
}  // namespace torch_mlu
