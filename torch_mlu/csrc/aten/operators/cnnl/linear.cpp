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

at::Tensor cnnl_linear_forward(at::Tensor& quantify_input, const int in_pos,
                               const at::Tensor& quantify_weight, const int w_pos,
                               const at::Tensor& bias, const bool is_per_channel,
                               const at::Tensor& new_scale) {
  // view the input dims to 2
  auto i_dim = quantify_input.dim();
  auto i_size = quantify_input.sizes().vec();
  int dim_0 = 1;
  if (i_dim > 2) {
    for (int i = 0; i < i_dim - 1; i++) dim_0 *= i_size[i];
    quantify_input = cnnl_view(quantify_input, {dim_0, i_size[i_dim - 1]});
  }
  bool is_trans_input = false;
  bool is_trans_weight = true;
  auto output = cnnl_mm(quantify_input, in_pos, quantify_weight,
                        w_pos, is_trans_input, is_trans_weight);
  if (is_per_channel) output = output / new_scale.reshape({1, -1});
  if (bias.dim() != 0) {
    output = cnnl_add(output, bias, 1);
  }
  if (i_dim > 2) {
    auto o_size = output.sizes();
    auto o_dim = output.dim();
    std::vector<int64_t> output_size;
    for (int i = 0; i < i_dim - 1; i++) output_size.push_back(i_size[i]);
    output_size.push_back(o_size[1]);
    at::IntList output_size_(output_size);
    output = cnnl_view(output, output_size_);
  }
  return output;
}

at::Tensor cnnl_linear(const at::Tensor& input, const at::Tensor& weight,
                       const at::Tensor& bias, const at::Tensor& q_scale,
                       const at::Tensor& q_mode) {
  auto input_ = cnnl_contiguous(input, input.suggest_memory_format());
  auto weight_ = cnnl_contiguous(weight, weight.suggest_memory_format());
  TORCH_CHECK(weight_.dim() == 2, "we only support the  dim of weight is 2");
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
    bool is_per_channel = (quantized_mode > 2 && quantized_mode < 5) ? true: false;
    // In both per-tensor/per-channl quantization, the input always uses per-tensor quantization.
    auto q_scale_tmp1 = q_scale_tmp.to(at::kFloat);
    float* scale_data = static_cast<float *>(q_scale_tmp1.storage().data());
    auto input_pos = get_pos_from_scale_data(bw, scale_data[0]);
    auto quantify_input = cnnl_quantify_offline(input_, bw, input_pos);
    if (!is_per_channel) {
      // doing quantization for weight
      auto weight_pos = get_pos_from_scale_data(bw, scale_data[1]);
      auto quantify_weight = cnnl_quantify_offline(weight_, bw, weight_pos);
      at::Tensor undef_new_scale;
      auto output = cnnl_linear_forward(quantify_input, input_pos, quantify_weight, weight_pos,
                                        bias, false, undef_new_scale);
      return output;
    } else {
      // Because cnnl linear does not support per-channel quantization,
      // so we use cnnl ops to do quant/dequant manually.

      // achieve weight scale in each channel from parameter q_scale.
      auto weight_scale = cnnl_slice(q_scale, 0, 2, q_scale.numel(), 1);
      // this pack include: 0->quantized_weight, 1->new_position, 2->new_scale
      auto quantized_weight_pack = cnnl_quantify_per_channel(weight_, weight_scale, bw);

      // quantize_linear
      auto quantify_weight = std::move(std::get<0>(quantized_weight_pack));
      auto new_position_weight = std::move(std::get<1>(quantized_weight_pack));
      auto new_pos_tmp = new_position_weight.cpu().to(at::kFloat);
      auto new_position_data = static_cast<float*>(new_pos_tmp.storage().data());
      auto new_scale_weight = std::move(std::get<2>(quantized_weight_pack));
      auto output = cnnl_linear_forward(quantify_input, input_pos, quantify_weight,
                                        (int)new_position_data[0], bias, true, new_scale_weight);
      return output;
    }
  } else {
    // In CNNL quantization interface:
    // * q_scale stores 2 values: input_bitwidth & weight_bitwidth;
    // * q_mode stores 2 values: input_position & weight_position.
    auto bw = static_cast<float*>(q_scale_tmp.storage().data());
    auto pos = static_cast<int*>(q_mode_tmp.storage().data());
    auto quantify_input = cnnl_quantify_offline(input_, int(bw[0]), pos[0]);

    auto quantify_weight = cnnl_quantify_offline(weight_, int(bw[1]), pos[1]);

    at::Tensor output = at::empty({0}, input_.options());

    // view the input dims to 2
    auto i_dim = input_.dim();
    auto i_size = input_.sizes().vec();
    int dim_0 = 1;
    if (i_dim > 2) {
      for (int i = 0; i < i_dim - 1; i++) dim_0 *= i_size[i];
      quantify_input = cnnl_view(quantify_input, {dim_0, i_size[i_dim - 1]});
    }
    bool is_trans_input = false;
    bool is_trans_weight = true;
    output = cnnl_mm(quantify_input, pos[0], quantify_weight,
                     pos[1], is_trans_input, is_trans_weight);
    if (bias.dim() != 0) {
      output = cnnl_add(output, bias, 1);
    }
    if (i_dim > 2) {
      auto o_size = output.sizes();
      auto o_dim = output.dim();
      std::vector<int64_t> output_size;
      for (int i = 0; i < i_dim - 1; i++) output_size.push_back(i_size[i]);
      output_size.push_back(o_size[1]);
      at::IntList output_size_(output_size);
      output = cnnl_view(output, output_size_);
    }
    return output;
  }
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
