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

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

static std::vector<int64_t> conv_output_size(at::IntList input_size,
                                             at::IntList weight_size,
                                             int *padding, int *stride,
                                             int *dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = weight_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

at::Tensor cnnl_float_convolution_internal(const at::Tensor & input,
           const at::Tensor & weight, const at::Tensor& bias, at::TensorOptions input_options,
           int* padding, int* stride, int* dilation, int64_t groups, bool depthwise) {
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto output = at::empty(conv_output_size(input.sizes(),
                                           weight.sizes(),
                                           padding,
                                           stride,
                                           dilation),
                          input_options,
                          memory_format);
  auto input_impl = getMluTensorImpl(input);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlTensorDescriptor output_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  input_desc.set(input, layout);
  auto weight_layout = depthwise ? CNNL_LAYOUT_HWCN : layout;
  weight_desc.set(weight, weight_layout);
  output_desc.set(output, layout);
  conv_desc.set(input.dim(), stride,
                padding, dilation, groups, getCnnlDataType(output.dtype()));

  // prepare conv desc
  cnnlConvolutionFwdPreference_t pre_t = CNNL_CONVOLUTION_FWD_FASTEST;
  cnnlConvolutionForwardAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), weight_desc.desc(),
      output_desc.desc(), pre_t, &algo_t));

  // prepare bias
  void *bias_ptr = nullptr;
  int64_t bias_size = 0;
  if (bias.defined() && bias.dim() != 0 && bias.numel() != 0) {
    TORCH_MLU_CHECK(bias.dim() == 1, "currently only support 1-dim bias in "
      "cnnl_float_convolution_internal when bias.dim() != 0, but got ", bias.dim(), " dim.");
    bias_size = bias.sizes()[0];
    // for group parameter, bias size must be 4 or 5 dims,(1,C,1,1) or (1,C,1,1,1)
    auto bias_impl = getMluTensorImpl(bias);
    layout = CNNL_LAYOUT_NHWC;
    resize_impl_mlu_(bias_impl, {1, bias_size, 1, 1}, c10::nullopt);
    bias_ptr = bias_impl->cnnlMalloc();
    bias_desc.set(bias, layout);
  }

  // prepare workspace
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
      handle, input_desc.desc(), weight_desc.desc(), output_desc.desc(),
      bias_desc.desc(), conv_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace = at::empty(workspace_size,
                          input.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  CnnlTransposeDescriptor trans_desc;
  at::Tensor weight_ = weight;
  if (depthwise) {
    weight_ = check_depth_transpose(weight, trans_desc, CNNL_LAYOUT_HWCN);
    weight_ptr = getMluTensorImpl(weight_)->cnnlMalloc();
  }

  const void * alpha = nullptr;
  const void * beta = nullptr;

  TORCH_CNNL_CHECK(cnnlConvolutionForward(
      /* handle         */ handle,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x_ptr          */ input_ptr,
      /* w_desc         */ weight_desc.desc(),
      /* w_ptr          */ weight_ptr,
      /* bias_desc      */ bias_desc.desc(),
      /* bias_ptr       */ bias_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y_ptr          */ output_ptr));
  if (bias.defined() && bias.dim() != 0) {
    resize_impl_mlu_(getMluTensorImpl(bias), {bias_size}, c10::nullopt);
  }
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
