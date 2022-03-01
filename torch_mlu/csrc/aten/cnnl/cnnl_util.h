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

#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input);

at::Tensor unsafe_trans_memory_format_inplace(const at::Tensor& input, bool channels_last = false);

std::vector<int64_t> get_trans_order(int64_t dim, bool channels_last = false);

at::Tensor cnnl_contiguous(const at::Tensor& input,
                           c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

at::MemoryFormat get_channels_last_memory_format(int64_t dim);

c10::MemoryFormat infer_memory_format(const int64_t& dims,
                                      const c10::MemoryFormat memory_format);

c10::MemoryFormat infer_tensor_list_contiguous(const at::TensorList& tensors);

bool is_permute(const at::Tensor& input);

std::vector<int64_t> get_permute_back_order(const at::Tensor& input);

std::vector<int64_t> get_permute_order(
        std::vector<int64_t> permute_back_order,
        c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

int get_quantized_bitwidth_from_input(const at::Tensor& input);

bool is_expand(const at::Tensor& input);

at::Tensor get_tensor_without_zero_stride(const at::Tensor& input);

at::Tensor non_overlapping_and_dense_out(at::Tensor& output, const at::Tensor& input);

std::vector<at::Tensor> unify_memory_format_of_multi_tensor(const at::TensorList& tensors);

at::Tensor permute_to_contiguous(const at::Tensor& input, c10::MemoryFormat memory_format);

int get_pos_from_scale_data(int bitwidth, float scale_data);

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
