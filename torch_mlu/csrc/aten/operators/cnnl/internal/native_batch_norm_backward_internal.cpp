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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_native_batch_norm_backward_internal(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean,
    const at::Tensor& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask) {
  auto common_options = weight.defined() ? weight.options() : input.options();
  TORCH_CHECK((input.dim() >= 2 && input.dim() <=5),
              "Input dim is out of range");

  auto weight_t = weight;
  int64_t c_dim = input.size(1);
  if (!weight_t.defined()) {
      weight_t = at::empty({c_dim}, common_options).fill_(1);
  }
  at::Tensor weight_bias_mean_var = at::empty({c_dim}, common_options);
  at::Tensor diff_weight = at::empty({c_dim}, common_options);
  at::Tensor diff_bias = at::empty({c_dim}, common_options);
  at::Tensor diff_x = at::empty_like(input);
  // A workaround if save_mean and save_invstd is not defined.
  at::Tensor save_mean_tensor = save_mean.defined() ? save_mean :
                                at::empty({c_dim}, common_options);
  at::Tensor save_invstd_tensor = save_invstd.defined() ? save_invstd :
                                  at::empty({c_dim}, common_options);

  auto input_impl = getMluTensorImpl(input);
  auto grad_impl = getMluTensorImpl(grad_out);
  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto weight_impl = getMluTensorImpl(weight_t);
  auto mean_impl = getMluTensorImpl(save_mean_tensor);
  auto var_impl = getMluTensorImpl(save_invstd_tensor);
  auto diff_weight_impl = getMluTensorImpl(diff_weight);
  auto diff_bias_impl = getMluTensorImpl(diff_bias);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlTensorDescriptor diff_x_desc;
  CnnlTensorDescriptor weight_bias_mean_var_desc;

  // When input is 2D or 3D Tensor, Reshape it to 4D tensor
  // NC -> NC11 -> NCHW | NCD -> NCD1 -> NCHW
  if (input.dim() == 3 || input.dim() == 2) {
    auto input_origin_shape = input.sizes().data();
    auto grad_origin_shape = grad_out.sizes().data();
    auto diff_x_origin_shape = diff_x.sizes().data();

    std::vector<int> input_shape{1, 1, 1, 1};
    std::vector<int> grad_shape{1, 1, 1, 1};
    std::vector<int> diff_x_shape{1, 1, 1, 1};
    for (int i = 0; i < input.dim(); ++i) {
      input_shape[i] = input_origin_shape[i];
      grad_shape[i] = grad_origin_shape[i];
      diff_x_shape[i] = diff_x_origin_shape[i];
    }
    // set cnnl descriptor
    input_desc.set_additional_dim(input, input_shape);
    grad_desc.set_additional_dim(grad_out, grad_shape);
    diff_x_desc.set_additional_dim(diff_x, diff_x_shape);
    weight_bias_mean_var_desc.set(weight_bias_mean_var);
  } else {
    cnnlTensorLayout_t layout =
        input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
    // set cnnl descriptor
    input_desc.set(input, layout);
    grad_desc.set(grad_out, layout);
    diff_x_desc.set(diff_x, layout);
    weight_bias_mean_var_desc.set(weight_bias_mean_var);
  }

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto diff_x_ptr = diff_x_impl->cnnlMalloc();
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto mean_ptr = mean_impl->cnnlMalloc();
  auto var_ptr = var_impl->cnnlMalloc();
  auto diff_weight_ptr = diff_weight_impl->cnnlMalloc();
  auto diff_bias_ptr = diff_bias_impl->cnnlMalloc();

  const void * alpha_data_diff = nullptr;
  const void * beta_data_diff = nullptr;
  const void * alpha_param_diff = nullptr;
  const void * beta_param_diff = nullptr;
  if (training) {
    TORCH_CNNL_CHECK(cnnlBatchNormBackward(
        /* handle           */ handle,
        /* alpha_data_diff  */ alpha_data_diff,
        /* beta_data_diff   */ beta_data_diff,
        /* alpha_param_diff */ alpha_param_diff,
        /* beta_param_diff  */ beta_param_diff,
        /* x_desc           */ input_desc.desc(),
        /* x                */ input_ptr,
        /* diff_z_desc      */ grad_desc.desc(),
        /* diff_z           */ grad_ptr,
        /* wbmv_desc        */ weight_bias_mean_var_desc.desc(),
        /* weight           */ weight_ptr,
        /* saved_mean       */ mean_ptr,
        /* saved_invstd     */ var_ptr,
        /* eps              */ static_cast<float>(eps),
        /* diff_x_desc      */ diff_x_desc.desc(),
        /* diff_x           */ diff_x_ptr,
        /* diff_weight      */ diff_weight_ptr,
        /* diff_bias        */ diff_bias_ptr));
  } else {
    auto running_mean_impl = getMluTensorImpl(running_mean);
    auto running_var_impl = getMluTensorImpl(running_var);
    auto running_mean_ptr = running_mean_impl->cnnlMalloc();
    auto running_var_ptr = running_var_impl->cnnlMalloc();

    TORCH_CNNL_CHECK(cnnlFrozenBatchNormBackward(
        /* handle      */ handle,
        /* x_desc      */ input_desc.desc(),
        /* x           */ input_ptr,
        /* diff_y_desc */ grad_desc.desc(),
        /* diff_y      */ grad_ptr,
        /* wbmv_desc   */ weight_bias_mean_var_desc.desc(),
        /* weight      */ weight_ptr,
        /* pop_mean    */ running_mean_ptr,
        /* pop_var     */ running_var_ptr,
        /* eps         */ static_cast<float>(eps),
        /* diff_x_desc */ diff_x_desc.desc(),
        /* diff_x      */ diff_x_ptr,
        /* diff_weight */ diff_weight_ptr,
        /* diff_bias   */ diff_bias_ptr));
  }
  return std::make_tuple(diff_x, diff_weight, diff_bias);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
