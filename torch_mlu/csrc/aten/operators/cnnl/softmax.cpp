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

at::Tensor cnnl_softmax_common(const at::Tensor& input, int64_t dim,
                         bool half_to_float, cnnlSoftmaxAlgorithm_t algo) {
  auto input_contiguous = cnnl_contiguous(input, input.suggest_memory_format());
  return cnnl_softmax_internal(input_contiguous,
                               dim, half_to_float, algo);
}

at::Tensor cnnl__softmax(const at::Tensor& input, int64_t dim,
                         bool half_to_float) {
  return cnnl_softmax_common(input, dim, half_to_float, CNNL_SOFTMAX_ACCURATE);
}

at::Tensor cnnl__softmax_backward_data(const at::Tensor& grad_output,
                                       const at::Tensor& output, int64_t dim,
                                       const at::Tensor& self) {
  auto memory_format = self.suggest_memory_format();
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  return cnnl_softmax_backward_internal(grad_output_contiguous,
                                        output_contiguous,
                                        dim,
                                        self_contiguous,
                                        CNNL_SOFTMAX_ACCURATE);
}

at::Tensor cnnl_log_softmax(const at::Tensor& input, int64_t dim,
                            bool half_to_float) {
  auto input_ = input;
  if (half_to_float) {
    TORCH_MLU_CHECK(input_.scalar_type() == at::ScalarType::Half,
      "conversion is supported for Half type only");
  }
  int64_t dim_ = at::maybe_wrap_dim(dim, input_.dim());
  return cnnl_softmax_common(input_, dim_, half_to_float, CNNL_SOFTMAX_LOG);
}

at::Tensor cnnl__log_softmax_backward_data(const at::Tensor& grad_output,
                                           const at::Tensor& output,
                                           int64_t dim,
                                           const at::Tensor& self) {
  auto memory_format = self.suggest_memory_format();
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  return cnnl_softmax_backward_internal(grad_output_contiguous,
                                        output_contiguous,
                                        dim,
                                        self_contiguous,
                                        CNNL_SOFTMAX_LOG);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
