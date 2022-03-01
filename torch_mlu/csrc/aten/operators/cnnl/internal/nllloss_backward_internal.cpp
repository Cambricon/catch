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
at::Tensor cnnl_nll_loss_backward_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) {
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
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto target_impl = getMluTensorImpl(target_cast);
  auto weight_impl = getMluTensorImpl(weight);
  auto tw_impl = getMluTensorImpl(total_weight);
  output_size = {N, C};
  cnnlNlllossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_REDUCTION_SUM;
      break;
    default:
      LOG(ERROR) << "nll_loss reduciton mode is avaliable";
      break;
  }
  // get current handle
  auto handle = getCurrentHandle();
  // prepare output, total_weight
  at::Tensor grad_input = at::empty(output_size, self.options(),
                                    c10::MemoryFormat::Contiguous);
  // deal with zero element tensor
  if (N == 0 || C == 0) {
    return grad_input;
  }
  auto grad_input_impl = getMluTensorImpl(grad_input);

  // get cnnl descriptor
  CnnlTensorDescriptor grad_output_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor tw_desc;
  CnnlTensorDescriptor grad_input_desc;
  grad_output_desc.set(grad_output);
  std::vector<int64_t> target_size({N});
  std::vector<int64_t> weight_size({C});
  std::vector<int64_t> weight_stride({1});
  target_desc.set(target_cast, target_size,
                  weight_stride, CNNL_LAYOUT_ARRAY);
  weight_desc.set(weight, weight_size,
                  weight_stride, CNNL_LAYOUT_ARRAY);
  grad_input_desc.set(grad_input);
  tw_desc.set(total_weight);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  auto target_ptr = target_impl->cnnlMalloc();
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto tw_ptr = tw_impl->cnnlMalloc();
  // calculate
  TORCH_CNNL_CHECK(cnnlNlllossBackward(
      handle, reduction_mode, grad_output_desc.desc(), grad_output_ptr,
      target_desc.desc(), target_ptr, ignore_idx, weight_desc.desc(),
      weight_ptr, tw_desc.desc(), tw_ptr, grad_input_desc.desc(),
      grad_input_ptr));
  return grad_input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
