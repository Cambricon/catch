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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/quantize_util.h"

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

at::Tensor cnnl_quantize_convolution_internal(const at::Tensor &input,
                                              const at::Tensor &input_position,
                                              const at::Tensor &weight,
                                              const at::Tensor &weight_position,
                                              const at::Tensor &bias,
                                              int *padding,
                                              int *stride,
                                              int *dilation,
                                              int64_t groups,
                                              int input_bitwidth,
                                              int weight_bitwidth,
                                              at::TensorOptions output_options) {
  auto input_onchip_dtype = get_onchip_dtype(input_bitwidth);
  auto weight_onchip_dtype = get_onchip_dtype(weight_bitwidth);
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto output = at::empty(conv_output_size(input.sizes(),
                                           weight.sizes(),
                                           padding,
                                           stride,
                                           dilation),
                                           output_options,
                                           memory_format);

  // TODO(liangyuefeng): Delete sclae in cnnl 1.2.1
  auto input_scale = at::empty(1, input.options().dtype(at::ScalarType::Float));
  auto weight_scale = at::empty(1, weight.options().dtype(at::ScalarType::Float));
  cnnl_fill_(input_scale, 1);
  cnnl_fill_(weight_scale, 1);

  auto input_impl = getMluTensorImpl(input);
  auto input_position_impl = getMluTensorImpl(input_position);
  auto input_scale_impl = getMluTensorImpl(input_scale);
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_position_impl = getMluTensorImpl(weight_position);
  auto weight_scale_impl = getMluTensorImpl(weight_scale);
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlTensorDescriptor output_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;

  // get current handle
  auto handle = getCurrentHandle();

  cnnlTensorLayout_t layout = input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;

  // realize pseudo conv3d by conv2d
  auto results = process_pseudo_conv(
      input, weight, output, &conv_desc,
      padding, stride, dilation, groups,
      getCnnlDataType(output.dtype()));
  bool pseudo_conv3d = std::get<3>(results);
  if (!pseudo_conv3d) {  // conv3d or conv2d
    input_desc.set(input, layout);
    weight_desc.set(weight, layout);
    output_desc.set(output, layout);
    conv_desc.set(input.dim(), stride, padding,
                  dilation, groups, getCnnlDataType(output.dtype()));
  } else {  // pseudo conv3d to conv2d
    auto input_size = std::get<0>(results);
    auto weight_size = std::get<1>(results);
    auto output_size = std::get<2>(results);
    TORCH_MLU_CHECK(input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    memory_format = c10::MemoryFormat::ChannelsLast;
    set_pseudo_conv_tensor_decs(input, input_size, layout, memory_format,
                                input_impl->getCnnlType(), input_desc);
    set_pseudo_conv_tensor_decs(weight, weight_size, layout, memory_format,
                                weight_impl->getCnnlType(), weight_desc);
    set_pseudo_conv_tensor_decs(output, output_size, layout, memory_format,
                                getCnnlDataType(output.dtype()), output_desc);
  }

  // prepare desc
  input_desc.set_onchip_dtype(input_onchip_dtype);
  weight_desc.set_onchip_dtype(weight_onchip_dtype);
  output_desc.set_onchip_dtype(getCnnlDataType(output.dtype()));

  // prepare conv desc
  cnnlConvolutionFwdPreference_t pre_t = CNNL_CONVOLUTION_FWD_FASTEST;
  cnnlConvolutionForwardAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), weight_desc.desc(),
      output_desc.desc(), pre_t, &algo_t));

  // prepare bias
  void *bias_ptr = nullptr;
  int64_t bias_size = 0;
  if (bias.defined() && bias.numel() != 0) {
    TORCH_MLU_CHECK(bias.dim() == 1, "currently only support 1-dim bias in "
      "cnnl_quantize_convolution_internal when bias.dim() != 0, but got ", bias.dim(), " dim.");
    bias_size = bias.sizes()[0];
    // for group parameter ,bias size must be 4 dims,(1,Co,1,1)
    auto bias_impl = getMluTensorImpl(bias);
    if (input.dim() > 4 && !pseudo_conv3d) {
      layout = CNNL_LAYOUT_NDHWC;
      resize_impl_mlu_(bias_impl, {1, bias_size, 1, 1, 1}, c10::nullopt);
    } else {
      layout = CNNL_LAYOUT_NHWC;
      resize_impl_mlu_(bias_impl, {1, bias_size, 1, 1}, c10::nullopt);
    }
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
  auto input_position_ptr = input_position_impl->cnnlMalloc();
  auto input_scale_ptr = input_scale_impl->cnnlMalloc();
  auto input_offset_ptr = nullptr;
  auto weight_ptr = weight_impl->cnnlMalloc();
  auto weight_position_ptr = weight_position_impl->cnnlMalloc();
  auto weight_scale_ptr = weight_scale_impl->cnnlMalloc();
  auto weight_offset_ptr = nullptr;
  auto output_ptr = output_impl->cnnlMalloc();

  const void * alpha = nullptr;
  const void * beta = nullptr;

  TORCH_CNNL_CHECK(cnnlQuantizeConvolutionForward(handle,
                                                  conv_desc.desc(),
                                                  algo_t,
                                                  alpha,
                                                  input_desc.desc(),
                                                  input_ptr,
                                                  input_position_ptr,
                                                  input_scale_ptr,
                                                  input_offset_ptr,
                                                  weight_desc.desc(),
                                                  weight_ptr,
                                                  weight_position_ptr,
                                                  weight_scale_ptr,
                                                  weight_offset_ptr,
                                                  bias_desc.desc(),
                                                  bias_ptr,
                                                  workspace_ptr,
                                                  workspace_size,
                                                  beta,
                                                  output_desc.desc(),
                                                  output_ptr));

  if (bias.defined() && bias.dim() != 0) {
    resize_impl_mlu_(getMluTensorImpl(bias), {bias_size}, c10::nullopt);
  }

  return output;
}

at::Tensor cnnl_quantize_convolution_internal(const at::Tensor &input,
                                              const at::Tensor &input_position,
                                              const at::Tensor &weight,
                                              const at::Tensor &weight_position,
                                              const at::Tensor &bias,
                                              int *padding,
                                              int *stride,
                                              int *dilation,
                                              int64_t groups,
                                              int bitwidth) {
  return cnnl_quantize_convolution_internal(input, input_position,
                                            weight, weight_position,
                                            bias, padding, stride,
                                            dilation, groups,
                                            bitwidth, bitwidth,
                                            input.options());
}

at::Tensor cnnl_quantize_convolution_backward_input_internal(const at::Tensor& input,
                                                             const at::Tensor& grad,
                                                             const at::Tensor& grad_position,
                                                             const at::Tensor& weight,
                                                             const at::Tensor& weight_position,
                                                             int* stride,
                                                             int* padding,
                                                             int* dilation,
                                                             int64_t groups,
                                                             int grad_bitwidth,
                                                             int weight_bitwidth,
                                                             at::TensorOptions output_options) {
  auto grad_onchip_dtype = get_onchip_dtype(grad_bitwidth);
  auto weight_onchip_dtype = get_onchip_dtype(weight_bitwidth);
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_input = at::empty(input.sizes(), output_options, memory_format);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_position_impl = getMluTensorImpl(weight_position);
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_position_impl = getMluTensorImpl(grad_position);
  CnnlTensorDescriptor grad_input_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;

  // get current handle
  auto handle = getCurrentHandle();

  cnnlTensorLayout_t layout = input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto results = process_pseudo_conv(
      grad_input, weight, grad, &conv_desc,
      padding, stride, dilation, groups,
      grad_input_impl->getCnnlType());
  bool pseduo_conv3d = std::get<3>(results);
  if (!pseduo_conv3d) {
    // Check the limit of parameters when dim is 5.
    if (input.dim() == 5) {
      TORCH_CHECK(dilation[0] == 1 && dilation[1] == 1 && dilation[2] == 1 &&
                  (groups == 1 || groups == weight.size(0)),
                  "unsupported parameters in conv bp data internel when dim is 5.");
    }
    grad_input_desc.set(grad_input, layout);
    weight_desc.set(weight, layout);
    grad_desc.set(grad, layout);
    conv_desc.set(input.dim(), stride, padding,
        dilation, groups, grad_input_impl->getCnnlType());
  } else {  // pseudo conv3d to conv2d
    auto grad_input_size = std::get<0>(results);
    auto weight_size = std::get<1>(results);
    auto grad_size = std::get<2>(results);
    TORCH_MLU_CHECK(grad_input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    memory_format = c10::MemoryFormat::ChannelsLast;
    set_pseudo_conv_tensor_decs(grad_input, grad_input_size, layout,
                                memory_format, grad_input_impl->getCnnlType(),
                                grad_input_desc);
    set_pseudo_conv_tensor_decs(weight, weight_size, layout,
                                memory_format, weight_impl->getCnnlType(),
                                weight_desc);
    set_pseudo_conv_tensor_decs(grad, grad_size, layout,
                                memory_format, grad_impl->getCnnlType(),
                                grad_desc);
  }

  // prepare desc
  weight_desc.set_onchip_dtype(weight_onchip_dtype);
  grad_desc.set_onchip_dtype(grad_onchip_dtype);

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
  auto weight_position_ptr = weight_position_impl->cnnlMalloc();
  auto weight_scale_ptr = nullptr;
  auto weight_offset_ptr = nullptr;
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto grad_position_ptr = grad_position_impl->cnnlMalloc();
  auto grad_scale_ptr = nullptr;
  auto grad_offset_ptr = nullptr;
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlQuantizeConvolutionBackwardData(handle,
                                                       alpha,
                                                       weight_desc.desc(),
                                                       weight_ptr,
                                                       weight_position_ptr,
                                                       weight_scale_ptr,
                                                       weight_offset_ptr,
                                                       grad_desc.desc(),
                                                       grad_ptr,
                                                       grad_position_ptr,
                                                       grad_scale_ptr,
                                                       grad_offset_ptr,
                                                       conv_desc.desc(),
                                                       algo_t,
                                                       workspace_ptr,
                                                       workspace_size,
                                                       beta,
                                                       grad_input_desc.desc(),
                                                       grad_input_ptr));
  return grad_input;
}

at::Tensor cnnl_quantize_convolution_backward_input_internal(const at::Tensor& input,
                                                             const at::Tensor& grad,
                                                             const at::Tensor& grad_position,
                                                             const at::Tensor& weight,
                                                             const at::Tensor& weight_position,
                                                             int* stride,
                                                             int* padding,
                                                             int* dilation,
                                                             int64_t groups,
                                                             int bitwidth) {
  return cnnl_quantize_convolution_backward_input_internal(input, grad, grad_position,
                                                           weight, weight_position, stride,
                                                           padding, dilation, groups,
                                                           bitwidth, bitwidth, input.options());
}

at::Tensor cnnl_quantize_convolution_backward_weight_internal(const at::Tensor& weight,
                                                              const at::Tensor& grad,
                                                              const at::Tensor& grad_position,
                                                              const at::Tensor& input,
                                                              const at::Tensor& input_position,
                                                              int* stride,
                                                              int* padding,
                                                              int* dilation,
                                                              int64_t groups,
                                                              int grad_bitwidth,
                                                              int input_bitwidth,
                                                              at::TensorOptions output_options) {
  auto grad_onchip_dtype = get_onchip_dtype(grad_bitwidth);
  auto input_onchip_dtype = get_onchip_dtype(input_bitwidth);
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_weight = at::empty(weight.sizes(), output_options, memory_format);
  auto grad_weight_impl = getMluTensorImpl(grad_weight);
  auto input_impl = getMluTensorImpl(input);
  auto input_position_impl = getMluTensorImpl(input_position);
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_position_impl = getMluTensorImpl(grad_position);
  CnnlTensorDescriptor grad_weight_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;

  // get current handle
  auto handle = getCurrentHandle();

  cnnlTensorLayout_t layout = input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto results = process_pseudo_conv(
      input, grad_weight, grad, &conv_desc,
      padding, stride, dilation, groups,
      grad_weight_impl->getCnnlType());
  bool pseudo_conv3d = std::get<3>(results);
  if (!pseudo_conv3d) {  // conv3d or conv2d
    input_desc.set(input, layout);
    grad_weight_desc.set(grad_weight, layout);
    grad_desc.set(grad, layout);
    conv_desc.set(input.dim(), stride, padding,
        dilation, groups, grad_weight_impl->getCnnlType());
  } else {  // pseudo conv3d to conv2d
    auto input_size = std::get<0>(results);
    auto grad_weight_size = std::get<1>(results);
    auto grad_size = std::get<2>(results);
    TORCH_MLU_CHECK(input_size.size() == 4, "conv2d only support 4 dims.");
    layout = CNNL_LAYOUT_NHWC;
    memory_format = c10::MemoryFormat::ChannelsLast;
    set_pseudo_conv_tensor_decs(input, input_size, layout, memory_format,
                                input_impl->getCnnlType(), input_desc);
    set_pseudo_conv_tensor_decs(grad_weight, grad_weight_size, layout,
                                memory_format, grad_weight_impl->getCnnlType(),
                                grad_weight_desc);
    set_pseudo_conv_tensor_decs(grad, grad_size, layout, memory_format,
                                grad_impl->getCnnlType(), grad_desc);
  }

  // prepare desc
  input_desc.set_onchip_dtype(input_onchip_dtype);
  grad_desc.set_onchip_dtype(grad_onchip_dtype);

  // prepare conv desc
  cnnlConvolutionBwdFilterPreference_t pre_t = CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
  cnnlConvolutionBwdFilterAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle, conv_desc.desc(), input_desc.desc(), grad_desc.desc(), grad_weight_desc.desc(),
      pre_t, &algo_t));

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
  auto input_position_ptr = input_position_impl->cnnlMalloc();
  auto input_scale_ptr = nullptr;
  auto input_offset_ptr = nullptr;
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto grad_position_ptr = grad_position_impl->cnnlMalloc();
  auto grad_scale_ptr = nullptr;
  auto grad_offset_ptr = nullptr;
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlQuantizeConvolutionBackwardFilter(handle,
                                                         alpha,
                                                         input_desc.desc(),
                                                         input_ptr,
                                                         input_position_ptr,
                                                         input_scale_ptr,
                                                         input_offset_ptr,
                                                         grad_desc.desc(),
                                                         grad_ptr,
                                                         grad_position_ptr,
                                                         grad_scale_ptr,
                                                         grad_offset_ptr,
                                                         conv_desc.desc(),
                                                         algo_t,
                                                         workspace_ptr,
                                                         workspace_size,
                                                         beta,
                                                         grad_weight_desc.desc(),
                                                         grad_weight_ptr));
  return grad_weight;
}

at::Tensor cnnl_quantize_convolution_backward_weight_internal(const at::Tensor& weight,
                                                              const at::Tensor& grad,
                                                              const at::Tensor& grad_position,
                                                              const at::Tensor& input,
                                                              const at::Tensor& input_position,
                                                              int* stride,
                                                              int* padding,
                                                              int* dilation,
                                                              int64_t groups,
                                                              int bitwidth) {
  return cnnl_quantize_convolution_backward_weight_internal(weight, grad, grad_position,
                                                            input, input_position, stride,
                                                            padding, dilation, groups,
                                                            bitwidth, bitwidth, weight.options());
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
