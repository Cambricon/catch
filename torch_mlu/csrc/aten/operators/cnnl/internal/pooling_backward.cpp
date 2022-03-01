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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "ATen/native/Pool.h"

using at::native::pooling_output_shape;
using at::native::avg_pool2d_backward_shape_check;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_avg_pool2d_backward_internal(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& index,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  bool isHalfData = (self.scalar_type() == at::kHalf);

  // output
  int64_t grad_c, grad_h, grad_w, batch_size;
  int64_t output_h, output_w;

  grad_w = grad.size(3);
  grad_h = grad.size(2);
  grad_c = grad.size(1);
  batch_size = grad.size(0);

  if (stride.size() == 0) {
    stride = kernel_size;
  }

  output_h = self.size(2);
  output_w = self.size(3);

  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size.size() == 1? kernel_h : kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride.size() == 1? stride_h: stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1? pad_h: padding[1];

  // calculate padding coefficients
  auto pl = 0, pr = 0, pu = 0, pd = 0;
  pu = pd = pad_h;
  pl = pr = pad_w;
  int height = (grad_h - 1) * stride_h + kernel_h;
  int width = (grad_w - 1) * stride_w + kernel_w;
  if (pad_h + output_h >= height) pd = 0;
  if (pad_w + output_w >= width) pr = 0;
  // if ceil_mode is set to true, the pad needs to be filled up.
  if (ceil_mode) {
    pd = height - output_h - pad_h;
    pr = width - output_w - pad_w;
  }

  std::vector<int64_t> output_size = {batch_size, grad_c, output_h, output_w};
  auto output = at::empty(output_size, grad.options(), at::MemoryFormat::ChannelsLast);
  auto temp = at::empty(grad.sizes().vec(), grad.options(), at::MemoryFormat::ChannelsLast);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;
  auto input_impl = getMluTensorImpl(self);
  auto grad_impl = getMluTensorImpl(grad);
  auto output_impl = getMluTensorImpl(output);

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  grad_desc.set(grad, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  index_desc.set(temp, isHalfData ? CNNL_DTYPE_INT16 : CNNL_DTYPE_INT32, CNNL_LAYOUT_NHWC);
  auto input_ptr = input_impl->cnnlMalloc();
  auto grad_ptr = grad_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  void* index_ptr = nullptr;

  // pooling mode
  cnnlPoolingMode_t mode = count_include_pad?
    CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
    CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  /* sizes */
  const int64_t nbatch = self.ndimension() == 4 ? self.size(-4) : 1;
  const int64_t nInputPlane = self.size(-3);  // number of channels (or colors)
  const int64_t inputHeight = self.size(-2);
  const int64_t inputWidth = self.size(-1);
  const int64_t outputWidth =
    pooling_output_shape<int64_t>(inputWidth, kernel_w, pad_w, stride_w, 1, ceil_mode);
  const int64_t outputHeight =
    pooling_output_shape<int64_t>(inputHeight, kernel_h, pad_h, stride_h, 1, ceil_mode);

  avg_pool2d_backward_shape_check(
      self,
      grad,
      nbatch,
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight, outputWidth);
  if (index.numel() > 0) {
    mode = CNNL_POOLING_MAX;
    auto index_impl = getMluTensorImpl(index);
    index_ptr = index_impl->cnnlMalloc();
  }
  const void * alpha = nullptr;
  const void * beta = nullptr;
  // PoolingBackward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, kernel_h, kernel_w, stride_h, stride_w, pu, pd, pl, pr);
  cnnlPoolingBackward(
      /* handle       */ handle,
      /* pooling_desc */ pooling_desc.desc(),
      /* alpha        */ alpha,
      /* y_desc       */ index_desc.desc(),
      /* y            */ index_ptr,
      /* diff_y_desc  */ grad_desc.desc(),
      /* diff_y       */ grad_ptr,
      /* x_desc       */ input_desc.desc(),
      /* x            */ input_ptr,
      /* beta         */ beta,
      /* diff_x_desc  */ output_desc.desc(),
      /* diff_x       */ output_ptr);

  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
