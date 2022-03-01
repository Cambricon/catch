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

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_batch_norm_internal(
    const at::Tensor& input, const at::Tensor& scale_tensor, const at::Tensor& bias_t,
    const at::Tensor& moving_mean, const at::Tensor& moving_var, bool training,
    double momentum, double eps) {
  TORCH_CHECK((input.dim() >= 2 && input.dim() <=5),
              "Input dim is out of range");
  auto common_options = scale_tensor.defined() ? scale_tensor.options() : input.options();

  int64_t c_dim = input.size(1);
  auto scale = scale_tensor;
  auto bias = bias_t;
  if (!scale.defined()) {
      scale = at::empty({c_dim}, common_options).fill_(1);
  }
  if (!bias.defined()) {
      bias = at::empty({c_dim}, common_options).fill_(0);
  }

  auto scale_bias_mean_var = at::empty({c_dim}, common_options);
  auto saved_mean = at::empty({c_dim}, common_options);
  auto saved_invstd = at::empty({c_dim}, common_options);
  at::Tensor output;

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor scale_bias_mean_var_desc;
  auto scale_bias_mean_var_layout = suggest_cnnl_layout(scale_bias_mean_var);
  scale_bias_mean_var_desc.set(scale_bias_mean_var, scale_bias_mean_var_layout);

  if (input.dim() == 3 || input.dim() == 2) {
    // Reshape 2D, 3D tensor to 4D tensor, thus can reuse bn2d as bn1d
    output = at::empty(input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
    auto input_origin_shape = input.sizes().data();
    auto output_origin_shape = output.sizes().data();
    std::vector<int> input_shape {1, 1, 1, 1};
    std::vector<int> output_shape {1, 1, 1, 1};
    for (int i = 0; i < input.dim(); i++) {
      input_shape[i] = input_origin_shape[i];
      output_shape[i] = output_origin_shape[i];
    }
    // get cnnl descriptor
    input_desc.set_additional_dim(input, input_shape);
    output_desc.set_additional_dim(output, output_shape);
  } else if (input.dim() == 4 || input.dim() == 5) {
    // get cnnl descriptor
    output = at::empty(input.sizes(), input.options(),
                       input.dim() == 4 ? c10::MemoryFormat::ChannelsLast
                       : c10::MemoryFormat::ChannelsLast3d);
    cnnlTensorLayout_t layout =
       input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
    input_desc.set(input, layout);
    output_desc.set(output, layout);
  }

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  auto saved_mean_impl = getMluTensorImpl(saved_mean);
  auto saved_invstd_impl = getMluTensorImpl(saved_invstd);

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  void* scale_ptr =
      scale.defined() ? getMluTensorImpl(scale)->cnnlMalloc() : nullptr;
  void* bias_ptr =
      bias.defined() ? getMluTensorImpl(bias)->cnnlMalloc() : nullptr;

  void* moving_mean_ptr = moving_mean.defined()
                              ? getMluTensorImpl(moving_mean)->cnnlMalloc()
                              : nullptr;
  void* moving_var_ptr = moving_var.defined()
                             ? getMluTensorImpl(moving_var)->cnnlMalloc()
                             : nullptr;

  auto saved_mean_ptr = saved_mean_impl->cnnlMalloc();
  auto saved_invstd_ptr = saved_invstd_impl->cnnlMalloc();
  const void* alpha = nullptr;
  const void* beta = nullptr;
  if (training) {
    TORCH_CNNL_CHECK(cnnlBatchNormForwardTraining(
        /* handle   */ handle,
        /* alpha    */ alpha,
        /* beta     */ beta,
        /* x_desc   */ input_desc.desc(),
        /* x        */ input_ptr,
        /* wbmvd    */ scale_bias_mean_var_desc.desc(),
        /* weight   */ scale_ptr,
        /* bias     */ bias_ptr,
        /* mov_mean */ moving_mean_ptr,
        /* mov_var  */ moving_var_ptr,
        /* eps      */ static_cast<float>(eps),
        /* momentum */ static_cast<float>(momentum),
        /* z_desc   */ output_desc.desc(),
        /* z        */ output_ptr,
        /* save_mean*/ saved_mean_ptr,
        /* save_std */ saved_invstd_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlBatchNormForwardInference(
        /* handle   */ handle,
        /* alpha    */ alpha,
        /* beta     */ beta,
        /* x_desc   */ input_desc.desc(),
        /* x        */ input_ptr,
        /* wbmvd    */ scale_bias_mean_var_desc.desc(),
        /* weight   */ scale_ptr,
        /* bias     */ bias_ptr,
        /* mov_mean */ moving_mean_ptr,
        /* mov_var  */ moving_var_ptr,
        /* eps      */ static_cast<float>(eps),
        /* z_desc   */ output_desc.desc(),
        /* z        */ output_ptr));
  }

  return std::make_tuple(output, saved_mean, saved_invstd);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
