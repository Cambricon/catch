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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean,
    const at::Tensor& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask) {
  if (running_mean.defined() && running_var.defined()) {
    auto mean_st = running_mean.dtype();
    auto var_st = running_var.dtype();
    TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
  }

  auto input_ = input;
  auto grad_ = grad_out;
  auto dim = input.dim();
  if (3 == dim) {
    input_ = input_.unsqueeze(3);
    grad_ = grad_.unsqueeze(3);
  }
  if (2 == dim) {
    input_ = input_.unsqueeze(2);
    input_ = input_.unsqueeze(3);
    grad_ = grad_.unsqueeze(2);
    grad_ = grad_.unsqueeze(3);
  }

  auto memory_format = get_channels_last_memory_format(input_.dim());
  auto input_contiguous = cnnl_contiguous(input_, memory_format);
  auto grad_contiguous = cnnl_contiguous(grad_, memory_format);
  auto output = cnnl_native_batch_norm_backward_internal(grad_contiguous,
                                                         input_contiguous,
                                                         weight, running_mean,
                                                         running_var, save_mean,
                                                         save_invstd, training,
                                                         eps, output_mask);

  auto & grad_input = std::get<0>(output);
  if (3 == dim)
    grad_input = grad_input.squeeze(3);
  if (2 == dim) {
    grad_input = grad_input.squeeze(3);
    grad_input = grad_input.squeeze(2);
  }
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
