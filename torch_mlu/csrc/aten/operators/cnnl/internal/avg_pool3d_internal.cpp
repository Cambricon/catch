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

#include <ATen/native/Pool.h>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

using at::native::safe_downcast;
using at::native::pooling_output_shape;
using at::native::pool3d_shape_check;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_avg_pool3d_internal(const at::Tensor& input,
                                    at::IntArrayRef kernel_size,
                                    at::IntArrayRef stride,
                                    at::IntArrayRef padding, bool ceil_mode,
                                    bool count_include_pad,
                                    int64_t pool_mode_row) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
                     ? kT
                     : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
                     ? kT
                     : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
              "avg_pool3d: stride must be omitted, a single int, or a tuple of "
              "three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH =
      stride.empty() ? kH : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW =
      stride.empty() ? kW : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
              "non-empty 4D or 5D (batch mode) tensor expected for input");

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(input, nslices, kT, kH, kW, dT, dH, dW, padT, padH, padW,
                     1, 1, 1, itime, iheight, iwidth, otime, oheight, owidth,
                     /*check_input_size=*/true);

  const int dilation = 1;
  std::vector<int> kernel_vec = {kT, kH, kW};
  std::vector<int> stride_vec = {dT, dH, dW};
  std::vector<int> padding_vec = {padT, padT, padH, padH, padW, padW};
  std::vector<int> dilation_vec = {dilation, dilation, dilation};

  std::vector<int64_t> output_size = {nbatch, nslices, otime, oheight, owidth};
  auto output = at::empty(output_size, input.options(), at::MemoryFormat::ChannelsLast3d);
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(input, CNNL_LAYOUT_NDHWC);
  output_desc.set(output, CNNL_LAYOUT_NDHWC);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
                               ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                               : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, 5, kernel_vec.data(), stride_vec.data(),
                   padding_vec.data(), dilation_vec.data(), ceil_mode);

  void* alpha = nullptr;
  void* beta = nullptr;
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;

  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
