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

at::Tensor cnnl_quantify_offline(const at::Tensor& input, const int bitwidth,
                                 const int position) {
  auto input_contiguous = input;
  if (!input.is_non_overlapping_and_dense()) input_contiguous = input.contiguous();
  return cnnl_quantify_offline_internal(input_contiguous, bitwidth, position);
}

at::Tensor cnnl_quantify_offline(const at::Tensor& input,
                                 const int bitwidth,
                                 const at::Tensor& position) {
  auto input_contiguous = input;
  if (!input.is_non_overlapping_and_dense()) input_contiguous = input.contiguous();
  return cnnl_quantify_offline_internal(input_contiguous, bitwidth, position);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_quantify_per_channel(
  const at::Tensor& input, const at::Tensor& scale_data, const int bitwidth) {
  float qmin = -128.0, qmax = 127.0;
  if (bitwidth == 16) {
    qmin = -32768.0;
    qmax = 32767.0;
  }
  // "absmax_pc" means "absmax_per_channel"
  // absmax_pc = qmax / scales
  auto absmax_pc = qmax / scale_data;

  // log2(absmax)
  // Note: absmax_pc has two kinds of datatype: fp32/fp16 when doing inference.
  //       Because cnnl_log2_out op only support computation on fp32, So we need to do datatype cast
  //       before and after this op when doing inference on fp16 inputs.
  auto log2_absmax = at::empty_like(absmax_pc);
  if (at::isFloatingType(absmax_pc.scalar_type())) {
    cnnl_log2_out(log2_absmax, absmax_pc);
  } else {
    auto absmax_pc_f32 = at::empty_like(absmax_pc, absmax_pc.options().dtype(at::ScalarType::Float));
    auto log2_absmax_f32 = at::empty_like(absmax_pc, absmax_pc.options().dtype(at::ScalarType::Float));
    cnnl_cast(absmax_pc, absmax_pc_f32);
    cnnl_log2_out(log2_absmax_f32, absmax_pc_f32);
    cnnl_cast(log2_absmax_f32, log2_absmax);
  }

  // pos = floor(log2(absmax)) - (n - 2)
  auto pos_pc =  at::empty_like(log2_absmax);
  cnnl_floor_out(pos_pc, log2_absmax);
  pos_pc = pos_pc - (bitwidth - 2);

  // scale = 2^pos * (2^(n - 1) - 1) / absmax
  auto scale_pc = cnnl_pow(at::Scalar(2), pos_pc);
  scale_pc = scale_pc * (std::pow(2, bitwidth - 1) - 1) / absmax_pc;

  // new_pos = round((Max(pos) + Min(pos)) / 2)
  auto new_pos = (cnnl_max(pos_pc) + cnnl_min(pos_pc)) / 2;
  new_pos = cnnl_round(new_pos).reshape({-1});

  // new_scale = scale_pc * 2^(new_pos - pos_pc)
  auto new_scale = scale_pc * cnnl_pow(at::Scalar(2), new_pos - pos_pc);

  // scale_factor = 2^new_pos / new_scale
  auto scale_factor = cnnl_pow(at::Scalar(2), new_pos) / new_scale;

  // quantized_input = clamp(qmin, qmax, round(input/scale_factor))
  std::vector<int64_t> vtr_exp_sf(input.dim(), 1);
  vtr_exp_sf[0] = -1;
  auto exp_sf_shape = c10::IntArrayRef(vtr_exp_sf);
  auto tmp_out = cnnl_clamp(cnnl_round(input / scale_factor.reshape(exp_sf_shape)),
                            at::Scalar(qmin), at::Scalar(qmax));

  at::Tensor quantify_input;
  if (bitwidth == 8) {
    quantify_input = tmp_out.to(at::ScalarType::Char);
  } else {
    quantify_input = tmp_out.to(at::ScalarType::Short);
  }
  return std::make_tuple(quantify_input, new_pos, new_scale);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu