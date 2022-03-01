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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/util/exceptions.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {
std::tuple<at::Tensor, at::Tensor> cnnl_nll_loss_forward_internal(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) {
  auto input_size = self.sizes().vec();
  int C = input_size[1];
  int N = std::accumulate(input_size.begin(), input_size.end(),
    1, std::multiplies<int64_t>()) / C;
  TORCH_MLU_CHECK(N == target.numel(), "Target size need be equal as input N*H*W.");
  TORCH_MLU_CHECK(C == weight.numel(), "Weight size need be equal as input C.");
  int ignore_idx = static_cast<int>(ignore_index);
  std::vector<int64_t> output_size;
  auto target_cast = target;
  if (target.scalar_type() != at::ScalarType::Int &&
      target.scalar_type() != at::ScalarType::Long) {
    target_cast = target.to(at::ScalarType::Int);
  }
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target_cast);
  auto weight_impl = getMluTensorImpl(weight);
  cnnlNlllossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_REDUCTION_NONE;
      output_size = {N};
      break;
    case 1:
      reduction_mode = CNNL_REDUCTION_MEAN;
      output_size = {};
      break;
    case 2:
      reduction_mode = CNNL_REDUCTION_SUM;
      output_size = {};
      break;
    default:
      LOG(ERROR) << "nll_loss reduciton mode is avaliable";
      break;
  }
  // get current handle
  auto handle = getCurrentHandle();
  // prepare output, total_weight
  std::vector<int64_t> total_weight_size = {1};
  at::Tensor output = at::empty(output_size, self.options());
  at::Tensor total_weight = at::empty(total_weight_size, weight.options());
  auto output_impl = getMluTensorImpl(output);
  auto tw_impl = getMluTensorImpl(total_weight);

  // get cnnl descriptor
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor tw_desc;
  CnnlTensorDescriptor output_desc;
  std::vector<int64_t> cnnl_input_size({N, C});
  std::vector<int64_t> input_stride({C, 1});
  std::vector<int64_t> target_size({N});
  std::vector<int64_t> weight_size({C});
  std::vector<int64_t> weight_stride({1});
  self_desc.set(self, cnnl_input_size,
                input_stride, CNNL_LAYOUT_ARRAY);
  target_desc.set(target_cast, target_size,
                  weight_stride, CNNL_LAYOUT_ARRAY);
  weight_desc.set(weight, weight_size,
                  weight_stride, CNNL_LAYOUT_ARRAY);
  output_desc.set(output);
  tw_desc.set(total_weight);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto self_ptr = self_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto tw_ptr = tw_impl->cnnlMalloc();

  // prepare workspace
  size_t sz;
  TORCH_CNNL_CHECK(cnnlGetNlllossWorkspaceSize(handle, self_desc.desc(), &sz));
  at::Tensor workspace = at::zeros(sz,
      at::TensorOptions(at::ScalarType::Byte).device(at::Device(at::Device::Type::MLU)));
  auto ws_impl = getMluTensorImpl(workspace);
  auto ws_ptr = ws_impl->cnnlMalloc();

  // calculate
  TORCH_CNNL_CHECK(cnnlNlllossForward(
      handle, reduction_mode, ws_ptr, sz, self_desc.desc(), self_ptr,
      target_desc.desc(), target_ptr, ignore_idx, weight_desc.desc(),
      weight_ptr, tw_desc.desc(), tw_ptr, output_desc.desc(), output_ptr));
  return std::make_tuple(output, total_weight);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
