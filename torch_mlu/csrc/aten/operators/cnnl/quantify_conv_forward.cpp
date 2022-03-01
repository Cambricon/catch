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

bool is_padding_neg(std::vector<int64_t> padding) {
  bool is_non_neg = false;
  for (int p : padding) {
    is_non_neg |= (p < 0);
  }
  return is_non_neg;
}

bool is_stride_nonpos(std::vector<int64_t> stride) {
  bool is_nonpos = false;
  for (int s : stride) {
    is_nonpos |= (s <= 0);
  }
  return is_nonpos;
}

at::Tensor cnnl_conv2d(const at::Tensor& input, const at::Tensor& weight,
                       const at::Tensor& bias, torch::List<int64_t> padding,
                       torch::List<int64_t> stride, torch::List<int64_t> dilation,
                       int64_t groups, const at::Tensor& q_scale, const at::Tensor& q_mode) {
  TORCH_MLU_CHECK(input.dim() > 2, "Input's dim must greater than 2!");

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto weight_contiguous = cnnl_contiguous(weight, memory_format);

  auto dim = input_contiguous.dim() - 2;
  int padding_t[dim], stride_t[dim], dilation_t[dim];
  for (int i = 0; i < dim; i++) {
    padding_t[i] = padding[i];
    stride_t[i] = stride[i];
    dilation_t[i] = dilation[i];
  }

  auto q_scale_tmp = is_mlu(q_scale) ? q_scale.cpu() : q_scale;
  auto q_mode_tmp = is_mlu(q_mode) ? q_mode.cpu() : q_mode;

  if (q_mode.numel() == 1) {
    // In CNML compatitive quantization interface:
    // * q_scale stores several values: input scale data & weight scale datas (per-tensor or per-channel);
    // * q_mode stores 1 values: quantization mode.
    //
    // Here are the detailed info of these two params:
    // q_scale = qmax / absmax
    //
    // q_mode = 
    // 1->qint8 or 
    // 2->qint16 or
    // 3->qint8-per-channel or 
    // 4->int16-per-channel
    int quantized_mode = static_cast<int*>(q_mode_tmp.storage().data())[0];
    int bw = (quantized_mode == 1 || quantized_mode == 3) ? 8 : 16;
    bool is_per_channel = (quantized_mode > 2 && quantized_mode < 5) ? true : false;
    // In both per-tensor/per-channl quantization, the input always uses per-tensor quantization.
    auto q_scale_tmp1 = q_scale_tmp.to(at::kFloat);
    float* scale_data = static_cast<float *>(q_scale_tmp1.storage().data());
    auto input_pos = get_pos_from_scale_data(bw, scale_data[0]);
    auto quantify_input = cnnl_quantify_offline(input_contiguous, bw, input_pos);
    if (!is_per_channel) {
      // doing quantization for weight
      auto weight_pos = get_pos_from_scale_data(bw, scale_data[1]);
      auto quantify_weight = cnnl_quantify_offline(weight_contiguous, bw, weight_pos);
      // doing inference
      auto output = cnnl_quantify_convolution_internal(quantify_input, input_pos,
           quantify_weight, weight_pos, bias, input_contiguous.options(), padding_t,
           stride_t, dilation_t, groups);
      return output;
    } else {
    // Because cnnl conv does not support per-channel quantization,
    // so we use cnnl ops to do quant/dequant manually.

    // achieve weight scale in each channel from parameter q_scale.
    auto weight_scale = cnnl_slice(q_scale, 0, 2, q_scale.numel(), 1);
    // this pack include: 0->quantized_weight, 1->new_position, 2->new_scale
    auto quantized_weight_pack = cnnl_quantify_per_channel(weight_contiguous, weight_scale, bw);

    // quantize_convolution
    auto quantify_weight = std::move(std::get<0>(quantized_weight_pack));
    auto new_position_weight = std::move(std::get<1>(quantized_weight_pack));
    auto new_pos_tmp = new_position_weight.cpu().to(at::kFloat);
    auto new_position_data = static_cast<float*>(new_pos_tmp.storage().data());
    auto new_scale_weight = std::move(std::get<2>(quantized_weight_pack));
    at::Tensor undefined_bias;
    auto output = cnnl_quantify_convolution_internal(quantify_input, input_pos,
         quantify_weight, (int)new_position_data[0], undefined_bias, input_contiguous.options(),
         padding_t, stride_t, dilation_t, groups);
    // dequant the output of convolution
    output = output / new_scale_weight.reshape({1, -1, 1, 1});
    // add bias
    output = output + bias.reshape({1, -1, 1, 1});
    return output;
    }
  } else {
    // In CNNL quantization interface:
    // * q_scale stores 2 values: input_bitwidth & weight_bitwidth;
    // * q_mode stores 2 values: input_position & weight_position.
    auto bw = static_cast<float*>(q_scale_tmp.storage().data());
    auto pos = static_cast<int*>(q_mode_tmp.storage().data());
    auto quantify_input = cnnl_quantify_offline(input_contiguous, int(bw[0]), pos[0]);

    auto quantify_weight = cnnl_quantify_offline(weight_contiguous, int(bw[1]), pos[1]);

    auto out = cnnl_quantify_convolution_internal(
        quantify_input, q_mode_tmp[0], quantify_weight, q_mode_tmp[1], bias,
        input_contiguous.options(), padding_t, stride_t, dilation_t, groups);
    return std::get<0>(out);
  }
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu