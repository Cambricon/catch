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

// When indices are bool: the dim size of each indice equals to the
// corresponding dim size of input，the number of indices is less equal to input
// There are several modes supported by AdvancedIndex for bool indices:
//
// 1. a[b, c]，eg: a.shape = [m, m] b.shape = [m] c.shape=[m], the output.
// shape maximum is [m]。This situation is special，because the shapes of b and c
// must be mutually broadcastable.
// 2. a[b], eg: a.shape = [m, n] b.shape = [m, n] output.shape maximum is [m * n],
// or a.shape = [m, n], b.shape = [m], output.shape is [m, n]
// 3. a[b,:], eg: a.shape = [m, n], b.shape = [m], output.shape maximum is [m, n] (same as mode 2)
//
// When indices are long: each value of every indices
// must be less equal to the maximum of input dims
// and support negative number (The absolute number of indices value should also less equal!)
// There are several modes supported by AdvancedIndex for long indices:
//
// 1. a[b,c], eg. a.shape = [n,m], b.shape = [n',m'], c.shape = [n',m'], output.shape = [n',m']
// 2. a[b], eg. a.shape = [n,m,q], b.shape = [n',m'], output.shape = [n',m',m,q]
// 3. a[b,:], eg. a.shape = [n,m,q], b.shape = [n',m'], output.shape = [n',m',m,q] (same as mode 2)
// 4. a[b,:,c], this mode equals to a[b,c] plus a transpotation to input tensor:
// eg. a.shape = [n,m,q], b.shape = [n',q'], c.shape=[n',q'], output.shape = [n',q',m]
//
//

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

// compute the output shape and broadcast shape of Long indices
std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t>>
compute_shapes(const at::Tensor& self,
               const std::vector<at::Tensor>& indices) {
  std::vector<int64_t> output_dims;
  auto self_dims = self.sizes().vec();
  auto indices_num = indices.size();
  std::vector<int64_t> indice_size;
  int64_t indice_dim;
  at::TensorOptions option;

  // reshape size for defined indices
  std::vector<int64_t> def_reshape_size;

  // check if defined indice has been calculated
  bool has_defined = false;

  // calculate output dims for indices
  for (int64_t j = 0; j < indices_num; ++j) {
    if (indices[j].defined()) {
      if (!has_defined) {
        indice_size = indices[j].sizes().vec();
        indice_dim = indices[j].dim();
        option = indices[j].options();
        output_dims.insert(output_dims.end(), indice_size.begin(), indice_size.end());
        def_reshape_size.insert(def_reshape_size.end(), indice_size.begin(), indice_size.end());
        std::vector<int64_t> ones_vec(indice_size.size(), 1);
        has_defined = true;
      }
    } else {
      output_dims.emplace_back(self_dims[j]);
      def_reshape_size.emplace_back(1);
    }
  }
  return std::make_tuple(indice_dim, output_dims, def_reshape_size);
}

// compute the output shape and broadcast shape of Bool indices
std::vector<int64_t> compute_shapes(const at::Tensor& self,
                                    const at::Tensor& indice) {
    std::vector<int64_t> output_dims;
    auto self_size = self.sizes().vec();
    output_dims.emplace_back(indice.numel());
    if (indice.dim() != self.dim()) {
        for (int64_t i = indice.dim(); i < self.dim(); ++i) {
            output_dims.emplace_back(self_size[i]);
        }
    }
    return output_dims;
}

std::tuple<at::Tensor, std::vector<at::Tensor>> make_info(at::Tensor self, at::TensorList orig) {
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = at::native::expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = at::expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together");
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // transpose self and indices together so that they're adjacent at the front
  if (!at::native::hasContiguousSubspace(indices)) {
    std::tie(self, indices) = at::native::transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return std::make_tuple(self, indices);
}

at::Tensor cnnl_index(const at::Tensor & self, at::TensorList indices) {
  TORCH_CHECK_INDEX(indices.size() <= (size_t)self.dim(),
          "too many indices for tensor of dimension ", self.dim());
  at::Tensor self_expand;
  std::vector<at::Tensor> indices_expand;
  std::vector<int64_t> output_dims;
  // use bool to compute when indices dtype is bool for better performance.
  if (indices.size() == 1 && indices[0].scalar_type() == at::ScalarType::Bool
          && (indices[0].numel() != 0)) {
    self_expand = self;
    indices_expand.emplace_back(indices[0]);
    output_dims = compute_shapes(self, indices[0]);
  } else {
    std::tie(self_expand, indices_expand) = make_info(self, indices);
    // compute the output shape
    auto shape_tuple = compute_shapes(self_expand, indices_expand);
    output_dims = std::get<1>(shape_tuple);
  }
  auto self_expand_contiguous = self_expand.contiguous();
  auto output = cnnl_index_internal(self_expand_contiguous, indices_expand, output_dims);
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
