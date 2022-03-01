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
using at::native::pool2d_shape_check;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_pool2d_internal(const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                bool ceil_mode, bool count_include_pad,
                                int64_t pool_mode_row) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "kernel_size must either be a single int, or a tuple of two ints");
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
      "padding must either be a single int, or a tuple of two ints");
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
      "stride must either be omitted, a single int, or a tuple of two ints");

  // TODO(huangzhichao): cnnl kernel does not support ceil_mode is set to True
  // TORCH_CHECK(!(ceil_mode == count_include_pad && count_include_pad == true),
  //  "cnnl kernel does not ceil_mode and count_include_pad are both true");

  int64_t input_w = self.size(3);
  int64_t input_h = self.size(2);
  int64_t input_c = self.size(1);
  int64_t batch_size = self.size(0);

  if (stride.size() == 0) {
    stride = kernel_size;
  }

  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size.size() == 1? kernel_h : kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride.size() == 1? stride_h: stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1? pad_h: padding[1];

  int64_t output_w = pooling_output_shape<int64_t>(input_w, kernel_w, pad_w,
                                           stride_w, 1, ceil_mode);
  int64_t output_h = pooling_output_shape<int64_t>(input_h, kernel_h, pad_h,
                                           stride_h, 1, ceil_mode);
  std::vector<int64_t> output_size = {batch_size, input_c, output_h, output_w};

  pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                     1, 1, input_c, input_h, input_w, output_w, output_h);

  auto input_impl = getMluTensorImpl(self);
  auto output = at::empty(output_size, self.options(), at::MemoryFormat::ChannelsLast);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad?
    CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
    CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (pool_mode_row > 0) mode = CNNL_POOLING_MAX;

  // workspace
  size_t space_size;
  TORCH_CNNL_CHECK(cnnlGetPoolingWorkspaceSize(handle, mode, output_size[2],
                                               output_size[3], &space_size));
  space_size /= input_impl->itemsize();
  auto temp = at::empty({static_cast<int64_t>(space_size)}, self.options());
  auto temp_impl = getMluTensorImpl(temp);
  auto temp_ptr = temp_impl->cnnlMalloc();

  // calculate padding coefficients
  auto pl = 0, pr = 0, pu = 0, pd = 0;
  pu = pd = pad_h;
  pl = pr = pad_w;
  if (ceil_mode) {
    // diff = (out - 1) * stride + kernel_size - input
    int diff_height = (output_h - 1) * stride_h + kernel_h - input_h;
    int diff_width = (output_w - 1) * stride_w + kernel_w - input_w;
    // If ceil_mode is set to true, the pad needs to be filled up.
    // If the offset pad is redundant, it will be removed.
    pd = diff_height > pad_h? diff_height - pad_h: 0;
    pr = diff_width > pad_w? diff_width - pad_w: 0;
  }
  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, kernel_h, kernel_w, stride_h, stride_w, pu, pd, pl, pr);
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* workspace      */ temp_ptr,
      /* workspace_size */ space_size * input_impl->itemsize()));
  return output;
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool_internal(const at::Tensor& self,
                                                          at::IntArrayRef kernel_size,
                                                          at::IntArrayRef stride,
                                                          at::IntArrayRef padding,
                                                          bool ceil_mode) {
  int64_t input_w = self.size(3);
  int64_t input_h = self.size(2);
  int64_t input_c = self.size(1);
  int64_t batch_size = self.size(0);

  if (stride.size() == 0) {
    stride = kernel_size;
  }

  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size.size() == 1? kernel_h : kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride.size() == 1? stride_h: stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1? pad_h: padding[1];

  int64_t output_w = pooling_output_shape<int64_t>(input_w, kernel_w, pad_w,
          stride_w, 1, ceil_mode);
  int64_t output_h = pooling_output_shape<int64_t>(input_h, kernel_h, pad_h,
          stride_h, 1, ceil_mode);
  std::vector<int64_t> output_size = {batch_size, input_c, output_h, output_w};

  pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                     1, 1, input_c, input_h, input_w, output_w, output_h);
  auto input_impl = getMluTensorImpl(self);
  auto output = at::empty(output_size, self.options(), at::MemoryFormat::ChannelsLast);
  auto index = at::empty(output_size,
                         self.options().dtype(caffe2::TypeMeta::Make<long>()),
                         at::MemoryFormat::ChannelsLast);
  auto output_impl = getMluTensorImpl(output);
  auto index_impl = getMluTensorImpl(index);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  cnnlDataType_t index_dtype = CNNL_DTYPE_INT32;
  if (CNNL_DTYPE_HALF == getCnnlDataType(self.dtype())) {
    index_dtype = CNNL_DTYPE_INT16;
  }
  index_desc.set(index, CNNL_LAYOUT_NHWC, index_dtype);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto index_ptr = index_impl->cnnlMalloc();


  // Determine the pooling mode
  cnnlPoolingMode_t mode = CNNL_POOLING_MAX;

  // workspace
  size_t space_size;
  TORCH_CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize(handle, input_desc.desc(),
                                                        output_desc.desc(), &space_size));
  std::vector<int64_t> space_shape;
  space_size /= input_impl->itemsize();
  space_shape.push_back(space_size);
  auto temp = at::empty(space_shape, self.options());
  auto temp_impl = getMluTensorImpl(temp);
  auto temp_ptr = temp_impl->cnnlMalloc();

  // calculate padding coefficients
  auto pl = 0, pr = 0, pu = 0, pd = 0;
  pu = pd = pad_h;
  pl = pr = pad_w;
  if (ceil_mode) {
    // diff = (out - 1) * stride + kernel_size - input
    int diff_height = (output_h - 1) * stride_h + kernel_h - input_h;
    int diff_width = (output_w - 1) * stride_w + kernel_w - input_w;
    // If ceil_mode is set to true, the pad needs to be filled up.
    // If the offset pad is redundant, it will be removed.
    pd = diff_height > pad_h? diff_height - pad_h: 0;
    pr = diff_width > pad_w? diff_width - pad_w: 0;
  }
  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, kernel_h, kernel_w, stride_h, stride_w, pu, pd, pl, pr);
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForwardWithIndex(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* index_desc     */ index_desc.desc(),
      /* index          */ index_ptr,
      /* workspace      */ temp_ptr,
      /* workspace_size */ space_size * input_impl->itemsize()));
  return std::make_tuple(output, index);
}

at::Tensor cnnl_pool3d_internal(const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                bool ceil_mode, bool count_include_pad,
                                int64_t pool_mode_row) {
  int64_t input_w = self.size(4);
  int64_t input_h = self.size(3);
  int64_t input_d = self.size(2);
  int64_t input_c = self.size(1);
  int64_t batch_size = self.size(0);

  if (stride.size() == 0) {
    stride = kernel_size;
  }

  int arrKernel[3], arrStride[3], arrPadding[6];
  arrKernel[0] = kernel_size[0];
  arrKernel[1] = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[1];
  arrKernel[2] = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[2];
  arrStride[0] = stride[0];
  arrStride[1] = stride.size() == 1 ? stride[0] : stride[1];
  arrStride[2] = stride.size() == 1 ? stride[0] : stride[2];
  arrPadding[1] = arrPadding[0] = padding[0];
  arrPadding[3] = arrPadding[2] = padding.size() == 1 ? padding[0] : padding[1];
  arrPadding[5] = arrPadding[4] = padding.size() == 1 ? padding[0] : padding[2];

  int64_t output_d = pooling_output_shape<int64_t>(
      input_d, arrKernel[0], arrPadding[0], arrStride[0], 1, ceil_mode);
  int64_t output_h = pooling_output_shape<int64_t>(
      input_h, arrKernel[1], arrPadding[2], arrStride[1], 1, ceil_mode);
  int64_t output_w = pooling_output_shape<int64_t>(
      input_w, arrKernel[2], arrPadding[4], arrStride[2], 1, ceil_mode);
  std::vector<int64_t> output_size = {
      batch_size, input_c, output_d, output_h, output_w};

  auto input_impl = getMluTensorImpl(self);
  auto output = at::empty(output_size, self.options(), at::MemoryFormat::ChannelsLast3d);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NDHWC);
  output_desc.set(output, CNNL_LAYOUT_NDHWC);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (pool_mode_row > 0)
    mode = CNNL_POOLING_MAX;

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, self.dim(), arrKernel, arrStride, arrPadding);
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
