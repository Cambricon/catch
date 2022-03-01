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

#pragma once

#include "aten/core/tensor_impl.h"
#include "aten/util/exceptions.h"
#include "aten/device/queue.h"
#include "aten/util/types.h"
#include "aten/util/cnlog.h"
#include "aten/util/common.h"
#include "aten/util/memory_allocator.h"

#include "aten/cnnl/cnnlDescriptors.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnl_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

void cnnl_cast(const at::Tensor& input, at::Tensor& output);
at::Tensor cnnl_mm(const at::Tensor& self, const at::Tensor& self_position,
                   const at::Tensor& other, const at::Tensor& other_position,
                   bool is_trans_a = false, bool is_trans_b = false);
at::Tensor cnnl_mm(const at::Tensor& self, const int self_position,
                   const at::Tensor& other, const int other_position,
                   bool is_trans_a = false, bool is_trans_b = false);
at::Tensor cnnl_quantify_offline(const at::Tensor& input, const int bitwidth,
                                 const int position);
at::Tensor cnnl_quantify_offline(const at::Tensor& input,
                                 const int bitwidth,
                                 const at::Tensor& position);
at::Tensor cnnl_bias_backward(const at::Tensor& self, int64_t dim);
at::Tensor cnnl_depthwise_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    torch::List<int64_t> padding, torch::List<int64_t> stride,
    torch::List<int64_t> dilation, int64_t groups);
at::Tensor cnnl_depthwise_convolution_backward_weight_internal(
    const at::Tensor& weight, const at::Tensor& grad, int grad_position_value,
    const at::Tensor& input, cnnlDataType_t input_dtype,
    int input_position_value, int* stride, int* padding, int* dilation,
    int64_t groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_quantify_per_channel(
    const at::Tensor& input, const at::Tensor& data_scale, const int bitwidth);

${cnnl_kernel_declarations}  // NOLINT

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
