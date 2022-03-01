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

using namespace torch::jit;

void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph) {
  // Remove all kinds of droupout from graph,
  // such as : dropout, dropout_, feature_droput, alpha_dropout, et.al.
  std::string GLOBAL_NO_DROPOUT_SIG = R"IR(
        graph(%input, %4, %5):
            return (%input))IR";


  std::string dropout_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::dropout(%input, %4, %5)
            return (%6))IR";
  std::string no_dropout_signature = R"IR(
        graph(%input, %4, %5):
            return (%input))IR";
  subgraph_rewrite(dropout_signature, GLOBAL_NO_DROPOUT_SIG, graph);

  std::string dropout_inplace_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::dropout_(%input, %4, %5)
            return (%6))IR";
  std::string no_dropout_inplace_signature = R"IR(
        graph(%input, %4, %5):
            return (%input))IR";
  subgraph_rewrite(dropout_inplace_signature, GLOBAL_NO_DROPOUT_SIG, graph);

  std::string feature_dropout_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::feature_dropout(%input, %4, %5)
            return (%6))IR";
  std::string no_feature_dropout_signature = R"IR(
        graph(%input, %4, %5):
            return (%input))IR";
  subgraph_rewrite(feature_dropout_signature, GLOBAL_NO_DROPOUT_SIG, graph);

  std::string feature_dropout_inplace_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::feature_dropout_(%input, %4, %5)
            return (%6))IR";
  subgraph_rewrite(feature_dropout_inplace_signature, GLOBAL_NO_DROPOUT_SIG, graph);

  std::string feature_alpha_dropout_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::feature_alpha_dropout(%input, %4, %5)
            return (%6))IR";

  subgraph_rewrite(feature_alpha_dropout_signature, GLOBAL_NO_DROPOUT_SIG, graph);

  // remove feature_alpha_dropout inplace
  std::string feature_alpha_dropout_inplace_signature = R"IR(
        graph(%input, %4, %5):
            %6 = aten::feature_alpha_dropout_(%input, %4, %5)
            return (%6))IR";

  subgraph_rewrite(feature_alpha_dropout_inplace_signature, GLOBAL_NO_DROPOUT_SIG, graph);
}

}  // namespace passes
}  // namespace lower
}  // namespace jit
}  // namespace torch_mlu
