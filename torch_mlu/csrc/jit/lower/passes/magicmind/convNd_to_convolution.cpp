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

void Conv2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string conv2d_signature = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %4 : Tensor = aten::conv2d(%x, %w, %b, %s, %p, %d, %g)
            return (%4))IR";
  std::string convolution_signature = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0, 0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";

  subgraph_rewrite(conv2d_signature, convolution_signature, graph);
}

void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string conv3d_signature = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %4 : Tensor = aten::conv3d(%x, %w, %b, %s, %p, %d, %g)
            return (%4))IR";
  std::string convolution_signature = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0, 0, 0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";

  subgraph_rewrite(conv3d_signature, convolution_signature, graph);
}


}  // namespace passes
}  // namespace lower
}  // namespace jit
}  // namespace torch_mlu
