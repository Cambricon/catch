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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::vector<int64_t> inferSqueezeSize(const at::Tensor& tensor) {
  std::vector<int64_t> sizes;
  for (int64_t d = 0; d < tensor.dim(); d++) {
    if (tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
    }
  }
  return sizes;
}

std::vector<int64_t> inferSqueezeSize(const at::Tensor& tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  for (int64_t d = 0; d < tensor.dim(); d++) {
    if (d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
    }
  }
  return sizes;
}

at::Tensor cnnl_squeeze(const at::Tensor& input) {
  auto* input_impl = getMluTensorImpl(input);
  auto output = at::native::squeeze(input);
  auto* output_impl = getMluTensorImpl(output);
  output_impl->insert_views_op_info(VIEWOPNAME::squeeze, input_impl,
                                    input.is_contiguous());
  return output;
}

at::Tensor cnnl_squeeze(const at::Tensor& input, int64_t dim) {
  auto* input_impl = getMluTensorImpl(input);
  auto output = at::native::squeeze(input, dim);
  auto* output_impl = getMluTensorImpl(output);
  output_impl->insert_views_op_info(VIEWOPNAME::squeeze, input_impl,
                                    input.is_contiguous());
  return output;
}

at::Tensor& cnnl_squeeze_(at::Tensor& input) {
  auto input_info = std::make_tuple(input.sizes().vec(),
                                    input.strides().vec(),
                                    input.storage_offset());
  bool input_contiguous = input.is_contiguous();
  at::native::squeeze_(input);
  auto* input_impl = getMluTensorImpl(input);
  input_impl->insert_views_op_info(VIEWOPNAME::squeeze,
                                   input_info,
                                   input_contiguous);
  return input;
}

at::Tensor& cnnl_squeeze_(at::Tensor& input, int64_t dim) {
  auto input_info = std::make_tuple(input.sizes().vec(),
                                    input.strides().vec(),
                                    input.storage_offset());
  bool input_contiguous = input.is_contiguous();
  at::native::squeeze_(input, dim);
  auto* input_impl = getMluTensorImpl(input);
  input_impl->insert_views_op_info(VIEWOPNAME::squeeze,
                                   input_info,
                                   input_contiguous);
  return input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
