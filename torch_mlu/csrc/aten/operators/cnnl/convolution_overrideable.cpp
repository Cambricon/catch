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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

namespace {
inline bool is_depthwise_conv(int64_t groups, int64_t ci, int64_t co, int64_t inc,
                              bool transposed) {
  // Input:  InN, Inc, InH, InW
  // Weight: Co,  Ci,  KH,  KW
  // Output: OutN,  OutC,  OutH,  OutW  # here OutC == Co
  // For DepthWise check, N,H,W are useless.
  // groups != 1, co % groups == 0, ci == 1 and groups == inc
  if (transposed) {
    //  the shape of in and out is transposed.
    //  Outc is inc, thus Outc == Co is co == inc here.
    return (groups != 1) && (co % groups == 0) && (ci == 1) && (inc == co);
  }
  return   (groups != 1) && (co % groups == 0) && (ci == 1) && (groups == inc);
}
}  // namespace

at::Tensor cnnl_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  auto dim = input.dim() - 2;
  std::vector<int> padding_t(dim, 0);
  std::vector<int> stride_t(dim, 0);
  std::vector<int> dilation_t(dim, 0);
  std::vector<int> output_padding_t(dim, 0);
  for (int i = 0; i < dim; i++) {
    padding_t[i] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
    output_padding_t[i] = output_padding[i];
  }

  if (Global::instance().isUsingFloatingDevice()) {
    if (transposed) {
      auto shape_tensor = at::empty(infer_deconv_output_size(input.sizes(),
                                weight.sizes(), padding_t.data(), output_padding_t.data(),
                                stride_t.data(), dilation_t.data(), groups),
                          input.options(), input.suggest_memory_format());

      if (bias.defined() && bias.numel() !=0) {
        std::vector<int64_t> vtr_exp_bias(input.dim(), 1);
        vtr_exp_bias[1] = shape_tensor.size(1);
        auto expand_bias = cnnl_contiguous(cnnl_view(bias, c10::IntArrayRef(vtr_exp_bias)),
                                           input.suggest_memory_format());

        return cnnl_add(cnnl_float_convolution_backward_input_internal(
                     shape_tensor, input, weight, stride_t.data(), padding_t.data(),
                     dilation_t.data(), groups, shape_tensor.options()), expand_bias, 1);
      } else {  // No bias add to conv
        return cnnl_float_convolution_backward_input_internal(
                     shape_tensor, input, weight, stride_t.data(), padding_t.data(),
                     dilation_t.data(), groups, shape_tensor.options());
      }
    } else {
      return cnnl_float_convolution_internal(input, weight, bias,
                                           input.options(), padding_t.data(),
                                           stride_t.data(), dilation_t.data(), groups);
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_convolution_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  auto dim = input.dim() - 2;
  int padding_t[dim], stride_t[dim], dilation_t[dim], output_padding_t[dim];
  for (int i = 0; i < dim; i++) {
    padding_t[i] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
    output_padding_t[i] = output_padding[i];
  }

  at::Tensor grad_input, grad_weight, grad_bias;
  if (Global::instance().isUsingFloatingDevice()) {
    if (output_mask[0]) {
      if (transposed) {
        auto bias = at::empty({weight.size(0)}, input.options());
        cnnl_zero_(bias);
        grad_input = cnnl_float_convolution_internal(grad, weight, bias, grad.options(),
                     padding_t, stride_t, dilation_t, groups);
      } else {
        grad_input = cnnl_float_convolution_backward_input_internal(
                    input, grad, weight, stride_t, padding_t, dilation_t,
                    groups, input.options());
      }
    }
    if (output_mask[1]) {
      if (transposed) {
        grad_weight = cnnl_float_convolution_backward_weight_internal(
                      weight, input, grad, stride_t, padding_t, dilation_t,
                      groups, weight.options());
      } else {
        grad_weight = cnnl_float_convolution_backward_weight_internal(
                      weight, grad, input, stride_t, padding_t, dilation_t,
                      groups, weight.options());
      }
    }
    if (output_mask[2]) {
      // compute bias grad in dim C
      int64_t dim = 1;
      grad_bias = cnnl_bias_backward_internal(grad, dim);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight,
                                                          grad_bias};
  }
}

at::Tensor cnnl_depthwise(const at::Tensor& input, const at::Tensor& weight,
                          const at::Tensor& bias, at::IntArrayRef padding,
                          at::IntArrayRef stride, at::IntArrayRef dilation,
                          int64_t groups, bool transposed, at::IntArrayRef output_padding) {
  TORCH_CHECK(input.dim() > 2, "Input's dim must greater than 2!");
  auto dim = input.dim() - 2;
  int padding_t[dim], stride_t[dim], dilation_t[dim], output_padding_t[dim];
  for (int i = 0; i < dim; i++) {
    padding_t[i] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
    output_padding_t[i] = output_padding[i];
  }
  if (Global::instance().isUsingFloatingDevice()) {
    if (transposed) {
      auto shape_tensor = at::empty(infer_deconv_output_size(input.sizes(),
                              weight.sizes(), padding_t, output_padding_t,
                              stride_t, dilation_t, groups), input.options(),
                              input.suggest_memory_format());
      auto weight_trans = weight;
      if (bias.defined() && bias.numel() !=0) {
        auto expand_bias = cnnl_contiguous(cnnl_view(bias, {1, shape_tensor.size(1), 1, 1}),
                                           input.suggest_memory_format());
        return cnnl_add(cnnl_float_convolution_backward_input_internal(
                                 shape_tensor, input,
                                 weight_trans, stride_t, padding_t, dilation_t, groups,
                                 shape_tensor.options(), true), expand_bias, 1);
      } else {
        return cnnl_float_convolution_backward_input_internal(
                                shape_tensor, input,
                                weight_trans, stride_t, padding_t, dilation_t, groups,
                                shape_tensor.options(), true);
      }
    } else {
      return cnnl_float_convolution_internal(input, weight, bias,
                                             input.options(), padding_t,
                                             stride_t, dilation_t, groups, /* depthwise */ true);
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_depthwise_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, std::array<bool, 3> output_mask, bool transposed,
    at::IntArrayRef output_padding) {
  TORCH_CHECK(input.dim() > 2, "Input's dim must greater than 2!");
  auto dim = input.dim() - 2;
  int padding_t[dim], stride_t[dim], dilation_t[dim];
  for (int i = 0; i < dim; i++) {
    padding_t[i] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
  }
  at::Tensor grad_input, grad_weight, grad_bias;
  if (Global::instance().isUsingFloatingDevice()) {
    if (output_mask[0]) {
      if (transposed) {
        auto bias = at::empty({weight.size(0)}, input.options());
        cnnl_zero_(bias);
        grad_input = cnnl_float_convolution_internal(grad, weight, bias, grad.options(),
                     padding_t, stride_t, dilation_t, groups, /* depthwise */ true);
      } else {
        grad_input = cnnl_float_convolution_backward_input_internal(
                    input, grad, weight, stride_t, padding_t, dilation_t,
                    groups, input.options(), /* depthwise */ true);
      }
    }
    if (output_mask[1]) {
      if (transposed) {
        grad_weight = cnnl_float_convolution_backward_weight_internal(
                      weight, input, grad, stride_t, padding_t, dilation_t,
                      groups, input.options(), /* depthwise */ true);
      } else {
        grad_weight = cnnl_float_convolution_backward_weight_internal(
                      weight, grad, input, stride_t, padding_t, dilation_t,
                      groups, weight.options(), /* depthwise */ true);
      }
    }
    if (output_mask[2]) {
      // compute bias grad in dim C
      int64_t dim = 1;
      grad_bias = cnnl_bias_backward_internal(grad, dim);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight,
                                                          grad_bias};
  }
}

std::set<at::ScalarType> conv_support_dtype{at::ScalarType::Half,
                                            at::ScalarType::Float,
                                            at::ScalarType::Double};

at::Tensor cnnl_convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  TORCH_MLU_CHECK(input.dim() > 2, "Input's dim must greater than 2!");
  TORCH_MLU_CHECK(conv_support_dtype.find(input.scalar_type()) != conv_support_dtype.end(),
                "Convolution mlu op not implemented for '", input.scalar_type(), "'");
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto weight_contiguous = cnnl_contiguous(weight, memory_format);
  if (is_depthwise_conv(groups, weight.size(1), weight.size(0), input.size(1), transposed)) {
    return cnnl_depthwise(input_contiguous,
                          weight_contiguous,
                          bias, padding,
                          stride, dilation,
                          groups, transposed, output_padding);
  }
  return cnnl_convolution(input_contiguous,
                          weight_contiguous,
                          bias, stride, padding,
                          dilation, transposed,
                          output_padding, groups);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_convolution_backward_overrideable(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  TORCH_MLU_CHECK(input.dim() > 2, "Input's dim must greater than 2!");
  TORCH_MLU_CHECK(conv_support_dtype.find(input.scalar_type()) != conv_support_dtype.end(),
                "Convolution mlu op not implemented for '", input.scalar_type(), "'");
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto weight_contiguous = cnnl_contiguous(weight, memory_format);
  auto grad_contiguous = cnnl_contiguous(grad, memory_format);
  if (is_depthwise_conv(groups, weight.size(1), weight.size(0), input.size(1), transposed)) {
    return cnnl_depthwise_backward(grad_contiguous,
                                   input_contiguous,
                                   weight_contiguous,
                                   padding, stride,
                                   dilation, groups,
                                   output_mask, transposed, output_padding);
  }
  return cnnl_convolution_backward(grad_contiguous,
                                   input_contiguous,
                                   weight_contiguous,
                                   stride, padding,
                                   dilation, transposed,
                                   output_padding, groups,
                                   output_mask);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
