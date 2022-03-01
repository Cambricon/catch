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

std::tuple<at::Tensor, at::Tensor> cnnl_maxpool3d_with_index_internal(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
              "max_pool3d: kernel_size must either be a single int, or a tuple "
              "of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
                     ? kT
                     : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
                     ? kT
                     : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
              "max_pool3d: stride must either be omitted, a single int, or a "
              "tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH =
      stride.empty() ? kH : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW =
      stride.empty() ? kW : stride.size() == 1
                                ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
              "max_pool3d: padding must be either be a single int, or a tuple "
              "of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
              "max_pool3d: dilation must be either a single int, or a tuple of "
              "three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
                            ? dilationT
                            : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
              "non-empty 4D or 5D (batch mode) tensor expected for input");

  const int64_t nbatch = self.ndimension() == 5 ? self.size(-5) : 1;
  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(self, nslices, kT, kH, kW, dT, dH, dW, pT, pH, pW,
                     dilationT, dilationH, dilationW, itime, iheight, iwidth,
                     otime, oheight, owidth);

  const int dim = 3;
  std::vector<int> kernel_vec = {kT, kH, kW};
  std::vector<int> stride_vec = {dT, dH, dW};
  std::vector<int> padding_vec = {pT, pT, pH, pH, pW, pW};
  std::vector<int> dilation_vec = {dilationT, dilationH, dilationW};

  std::vector<int64_t> output_size = {nbatch, nslices, otime, oheight, owidth};
  auto input_impl = getMluTensorImpl(self);
  auto output = at::empty(output_size, self.options(), at::MemoryFormat::ChannelsLast3d);
  auto output_impl = getMluTensorImpl(output);

  cnnlTensorLayout_t layout = CNNL_LAYOUT_NDHWC;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;
  input_desc.set(self, layout);
  output_desc.set(output, layout);
  at:Tensor index;
  if (CNNL_DTYPE_HALF == getCnnlDataType(self.dtype())) {
      index = at::empty(output_size,
                        self.options().dtype(caffe2::TypeMeta::Make<short>()),
                        at::MemoryFormat::ChannelsLast3d);
  } else if (CNNL_DTYPE_FLOAT == getCnnlDataType(self.dtype())) {
      index = at::empty(output_size,
                        self.options().dtype(caffe2::TypeMeta::Make<int>()),
                        at::MemoryFormat::ChannelsLast3d);
  }
  auto index_impl = getMluTensorImpl(index);
  index_desc.set(index, layout);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto index_ptr = index_impl->cnnlMalloc();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = CNNL_POOLING_MAX;

  // workspace
  size_t space_size;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize(
      handle, input_desc.desc(), output_desc.desc(), &space_size));
  std::vector<int64_t> space_shape;
  space_size /= input_impl->itemsize();
  space_shape.push_back(space_size);
  auto temp = at::empty(space_shape, self.options());
  auto temp_impl = getMluTensorImpl(temp);
  auto temp_ptr = temp_impl->cnnlMalloc();

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, 5, kernel_vec.data(), stride_vec.data(),
                   padding_vec.data(), dilation_vec.data(), ceil_mode);

  const void* alpha = nullptr;
  const void* beta = nullptr;

  TORCH_CNNL_CHECK(cnnlPoolingForwardWithIndex(
      /* handle */ handle,
      /* pooling_desc */ pooling_desc.desc(),
      /* alpha */ alpha,
      /* x_desc */ input_desc.desc(),
      /* x */ input_ptr,
      /* beta */ beta,
      /* y_desc */ output_desc.desc(),
      /* y */ output_ptr,
      /* index_desc */ index_desc.desc(),
      /* index */ index_ptr,
      /* workspace */ temp_ptr,
      /* workspace_size */ space_size * input_impl->itemsize()));
  return std::make_tuple(output, index);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
