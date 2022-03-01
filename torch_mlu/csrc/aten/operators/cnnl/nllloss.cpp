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
// nll_loss_wrapper
std::tuple<at::Tensor, at::Tensor> cnnl_nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) {
  TORCH_MLU_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_MLU_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_MLU_CHECK(
      self.size(0) == target.size(0),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")");

  const auto n_classes = self.size(-1);

  TORCH_MLU_CHECK(
      !weight.defined() || weight.numel() == n_classes,
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  bool is_half_data = (self.scalar_type() == at::kHalf);
  at::Tensor weight_;

  auto self_cast = is_half_data ? self.to(at::kFloat) : self;

  at::Tensor output;
  at::Tensor total_weight;
  if (!weight.defined()) {
    weight_ = at::ones(
        self_cast.size(1), self_cast.options().device(at::Device(at::Device::Type::MLU)));
  } else {
    weight_ = is_half_data ? weight.to(at::kFloat) : weight;
  }
  auto self_contiguous = cnnl_contiguous(self_cast);
  auto weight_contiguous = cnnl_contiguous(weight_);
  auto target_contiguous = cnnl_contiguous(target);
  std::tie(output, total_weight) = cnnl_nll_loss_forward_internal(
      self_contiguous, target_contiguous, weight_contiguous, reduction,
      ignore_index);
  auto output_cast = is_half_data ? output.to(at::kHalf) : output;
  auto total_weight_cast = is_half_data ? total_weight.to(at::kHalf) : total_weight;
  return std::make_tuple(output_cast, total_weight_cast);
}

std::tuple<at::Tensor, at::Tensor> cnnl_nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) {
  /*
   * transform nll_loss2d to nll_loss
   * ==> nll_loss2d
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   *      ||                            ||
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   *
   * ==> nll_loss
   */
  at::Tensor output;
  at::Tensor total_weight;
  at::Tensor weight_ = weight;
  if (!weight.defined()) {
    weight_ = at::ones(self.size(1),
            self.options().device(at::Device(at::Device::Type::MLU)));
  }
  auto weight_contiguous = cnnl_contiguous(weight_, weight_.suggest_memory_format());

  /*
   * transform self
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   */

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);

  /*
   * transform weight
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   * [m, d1, d2]    (permute 021)  [m, d1, d2]
   *
   * [mxd1xd2]                     [mxd1xd2]
   *
   */
  auto target_contiguous = cnnl_contiguous(target, target.suggest_memory_format());

  // nll_loss
  std::tie(output, total_weight) = cnnl_nll_loss_forward_internal(
      self_contiguous, target_contiguous, weight_contiguous, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
