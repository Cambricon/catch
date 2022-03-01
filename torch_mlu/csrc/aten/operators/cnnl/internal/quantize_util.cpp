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

#include "aten/operators/cnnl/internal/quantize_util.h"
#include <algorithm>

namespace torch_mlu {
namespace cnnl {
namespace ops {

cnnlDataType_t get_onchip_dtype(int bitwidth) {
  cnnlDataType_t onchip_dtype;
  switch (bitwidth) {
    case 8:
        onchip_dtype = CNNL_DTYPE_INT8;
        break;
    case 16:
        onchip_dtype = CNNL_DTYPE_INT16;
        break;
    case 31:
        onchip_dtype = CNNL_DTYPE_INT31;
        break;
    default:
        LOG(ERROR) << "bitwidth not support!";
        break;
  }
  return onchip_dtype;
}

// use conv2d to implement parts of con3d
std::tuple<std::vector<int64_t>, std::vector<int64_t>,
           std::vector<int64_t>, bool> process_pseudo_conv(
    const at::Tensor& gin,  // input or grad_input
    const at::Tensor& gw,  // weight or grad_weight
    const at::Tensor& gout,  // output or grad
    CnnlConvolutionDescriptor* conv_desc,
    int* padding,
    int* stride,
    int* dilation,
    int64_t groups,
    cnnlDataType_t data_type) {
  std::vector<int64_t> fake_gin;
  std::vector<int64_t> fake_gw;
  std::vector<int64_t> fake_gout;
  if (gin.dim() == 5 && gw.size(2) == 1 && stride[0] == 1 && padding[0] == 0 &&
      dilation[0] == 1) {
    auto combine_second_dims = [](const std::vector<int64_t>& input,
                                  std::vector<int64_t>& output) -> void {
      output.resize(4);
      output[0] = input[0] * input[2];
      output[1] = input[1];
      output[2] = input[3];
      output[3] = input[4];
    };
    combine_second_dims(gin.sizes().vec(), fake_gin);
    combine_second_dims(gw.sizes().vec(), fake_gw);
    combine_second_dims(gout.sizes().vec(), fake_gout);
    int padding_t[2] = {padding[1], padding[2]};
    int stride_t[2] = {stride[1], stride[2]};
    int dilation_t[2] = {dilation[1], dilation[2]};
    if (conv_desc) {
      conv_desc->set(4, stride_t, padding_t, dilation_t, groups, data_type);
    }
    return std::make_tuple(fake_gin, fake_gw, fake_gout, true);
  } else if (gin.dim() == 5 && (gw.size(3) == 1 && gw.size(4) == 1) &&
      (stride[1] == 1 && stride[2] == 1) &&
      (padding[1] == 0 && padding[2] == 0) &&
      (dilation[1] == 1 && dilation[2] == 1)) {
    auto combine_second_dims = [](const std::vector<int64_t>& input,
                                  std::vector<int64_t>& output) -> void {
      output.resize(4);
      output[0] = input[0];
      output[1] = input[1];
      output[2] = input[2];
      output[3] = input[3] * input[4];
    };
    combine_second_dims(gin.sizes().vec(), fake_gin);
    combine_second_dims(gw.sizes().vec(), fake_gw);
    combine_second_dims(gout.sizes().vec(), fake_gout);
    int padding_t[2] = {padding[0], 0};
    int stride_t[2] = {stride[0], 1};
    int dilation_t[2] = {dilation[0], 1};
    if (conv_desc) {
      conv_desc->set(4, stride_t, padding_t, dilation_t, groups, data_type);
    }
    return std::make_tuple(fake_gin, fake_gw, fake_gout, true);
  } else {
    return std::make_tuple(gin.sizes().vec(), gw.sizes().vec(), gout.sizes().vec(), false);
  }
}

void set_pseudo_conv_tensor_decs(const at::Tensor self,
                                 const std::vector<int64_t>& size,
                                 const cnnlTensorLayout_t layout,
                                 c10::MemoryFormat memory_format,
                                 cnnlDataType_t data_type,
                                 CnnlTensorDescriptor& self_desc) {
  auto self_size = modify_dims_based_on_layout(size, memory_format);
  auto self_stride = get_contiguous_strides(self_size);
  std::vector<int> int_size;
  std::vector<int> int_stride;
  auto convert_int64_int = [] (const std::vector<int64_t>& size,
                               std::vector<int>& output) -> void {
    const int dim = size.size();
    output.resize(dim);
    for (int i = 0; i < dim; i++) {
      output[i] = static_cast<int>(size[i]);
    }
  };
  convert_int64_int(self_size, int_size);
  convert_int64_int(self_stride, int_stride);
  self_desc.set(self, int_size, int_stride, layout, data_type);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
