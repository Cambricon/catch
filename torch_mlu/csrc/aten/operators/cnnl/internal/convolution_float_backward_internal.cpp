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

at::Tensor cnnl_float_convolution_backward_input_internal(
    const at::Tensor& input, const at::Tensor& grad,
    const at::Tensor& weight, int* stride,
    int* padding, int* dilation, int64_t groups, at::TensorOptions input_options,
    bool depthwise) {
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_input = at::empty(input.sizes(), input_options, memory_format);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto weight_impl = getMluTensorImpl(weight);
  auto grad_impl = getMluTensorImpl(grad);
  CnnlTensorDescriptor grad_input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  auto layout = CNNL_LAYOUT_NHWC;

  grad_input_desc.set(grad_input, layout);
  weight_desc.set(weight, layout);
  grad_desc.set(grad, layout);
  conv_desc.set(input.dim(), stride, padding,
                dilation, groups, getCnnlDataType(grad_input.dtype()));

  // prepare conv desc
  cnnlConvolutionBwdDataPreference_t pre_t = CNNL_CONVOLUTION_BWD_DATA_FASTEST;
  cnnlConvolutionBwdDataAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
      handle, weight_desc.desc(), grad_desc.desc(), conv_desc.desc(),
      grad_input_desc.desc(), pre_t, &algo_t));
  // prepare workspace
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle, weight_desc.desc(), grad_desc.desc(), conv_desc.desc(),
      grad_input_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, weight.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }
  // malloc mlu memory
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto grad_ptr = grad_impl->cnnlMalloc();
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlConvolutionBackwardData(
      /* handle         */ handle,
      /* alpha          */ alpha,
      /* weight_desc    */ weight_desc.desc(),
      /* weight         */ weight_ptr,
      /* diff_y_desc    */ grad_desc.desc(),
      /* diff_y         */ grad_ptr,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* diff_x_desc    */ grad_input_desc.desc(),
      /* diff_x         */ grad_input_ptr));
  return grad_input;
}

at::Tensor cnnl_float_convolution_backward_weight_internal(
    const at::Tensor& weight, const at::Tensor& grad,
    const at::Tensor& input, int* stride,
    int* padding, int* dilation, int64_t groups, at::TensorOptions weight_options,
    bool depthwise) {
  auto memory_format = get_channels_last_memory_format(weight.dim());
  auto grad_weight = at::empty(weight.sizes(), weight_options, memory_format);
  auto grad_weight_impl = getMluTensorImpl(grad_weight);
  auto input_impl = getMluTensorImpl(input);
  auto grad_impl = getMluTensorImpl(grad);
  CnnlTensorDescriptor grad_weight_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  auto layout = CNNL_LAYOUT_NHWC;
  input_desc.set(input, layout);
  auto weight_layout = depthwise ? CNNL_LAYOUT_HWCN : layout;
  grad_weight_desc.set(grad_weight, weight_layout);
  grad_desc.set(grad, layout);
  conv_desc.set(weight.dim(), stride, padding,
      dilation, groups, getCnnlDataType(grad_weight.dtype()));

  // prepare conv desc
  cnnlConvolutionBwdFilterPreference_t pre_t =
      CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
  cnnlConvolutionBwdFilterAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), grad_desc.desc(),
      grad_weight_desc.desc(), pre_t, &algo_t));
  // prepare workspace
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardFilterWorkspaceSize(
      handle, input_desc.desc(), grad_desc.desc(), grad_weight_desc.desc(),
      conv_desc.desc(), algo_t, &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, weight.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }
  // malloc mlu memory


  auto grad_weight_ptr = grad_weight_impl->cnnlMalloc();
  auto input_ptr = input_impl->cnnlMalloc();
  auto grad_ptr = grad_impl->cnnlMalloc();
  CnnlTransposeDescriptor trans_desc;

  const void * alpha = nullptr;
  const void * beta = nullptr;

  TORCH_CNNL_CHECK(cnnlConvolutionBackwardFilter(
      /* handle         */ handle,
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* diff_y_desc    */ grad_desc.desc(),
      /* diff_y         */ grad_ptr,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* diff_w_desc    */ grad_weight_desc.desc(),
      /* diff_w         */ grad_weight_ptr));
  if (depthwise) {
    return check_depth_transpose(grad_weight, trans_desc, CNNL_LAYOUT_NHWC);
  }
  return grad_weight;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
