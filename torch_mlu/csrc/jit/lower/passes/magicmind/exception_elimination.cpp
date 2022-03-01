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

#include "jit/lower/passes/magicmind/passes.h"
#include "jit/lower/passes/magicmind/subgraph_rewrite.h"

namespace torch_mlu {
namespace jit {
namespace lower {
namespace passes {

#define EXP_RET_BLOCK_SIZE 2
using namespace torch::jit;

struct RemoveExciptionOrPassOp {
  explicit RemoveExciptionOrPassOp(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findExceptionCheckOp(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
  }

 private:
  template<class T>
  inline bool checkBlocksOnlyHaveExceptionReturn(T node1, T node2) {
    if (node1->kind() == prim::RaiseException) {
      if ((++node1)->kind() != prim::Return) {
        // Check if current block only have exception check.
        return false;
      }
      if (node2->kind() != prim::Return) return false;
    }
    return true;
  }

  bool isExceptionOrPassNode(Node* n) {
    if (n->blocks().size() != EXP_RET_BLOCK_SIZE) {
      return false;
    }
    auto branch1 = n->blocks()[0];
    auto branch2 = n->blocks()[1];
    if (branch1->outputs().size() != 0 || branch2->outputs().size() != 0) {
      // Check if the current node have any output which will be used by other nodes.
      return false;
    }

    auto branch1_start = branch1->nodes().begin();
    auto branch2_start = branch2->nodes().begin();

    /// Check the senario of -> block0: prim::RaiseException() & block1: Return
    if ((*branch1_start)->kind() == prim::RaiseException)
      return checkBlocksOnlyHaveExceptionReturn(branch1_start, branch2_start);
    // vice versa ->  block0: prim::Return & block1: prim::RaiseException()
    else if ((*branch2_start)->kind() == prim::RaiseException)
      return checkBlocksOnlyHaveExceptionReturn(branch2_start, branch1_start);

    return true;
  }

  void findExceptionCheckOp(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::If && isExceptionOrPassNode(n)) {
        std::cout << "Remove the Exception/Pass node: " << *n << std::endl;
        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};

void RemoveExceptionCheck(std::shared_ptr<Graph> graph) {
  RemoveExciptionOrPassOp eppe(std::move(graph));
  eppe.run();
}

}  // namespace passes
}  // namespace lower
}  // namespace jit
}  // namespace torch_mlu

