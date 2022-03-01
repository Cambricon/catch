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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

using at::native::safe_downcast;
using at::native::pooling_output_shape;
using at::native::pool3d_shape_check;

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::set<at::ScalarType> Pool3dSupportDtype{at::ScalarType::Float, at::ScalarType::Half};

at::Tensor cnnl_avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           bool ceil_mode, bool count_include_pad,
                           c10::optional<int64_t> divisor_override) {
  TORCH_MLU_CHECK(self.ndimension() == 4,
                  "cnnl pool2d only support 4D input tensor currently.");
  TORCH_MLU_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
                  "divisor must be not zero");
  TORCH_MLU_CHECK(at::isFloatingType(self.scalar_type()),
              "avg_pool2d on mlu only support self scalar type Float and Half");
  if (self.numel() == 0) return self;

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  return cnnl_pool2d_internal(self_contiguous, kernel_size, stride, padding, ceil_mode,
                              count_include_pad, 0);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool2d_with_indices(
    const at::Tensor& input, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  TORCH_CHECK(input.ndimension() == 4,
              "cnnl pool2d only support 4D input tensor currently.");
  constexpr char dilation_err[] =
    "max_pool2d: dilation must be either a single int, or a tuple of two ints, "
    "and cnnl pool2d only supports defalut dilation value";
  TORCH_CHECK((dilation.size() == 1 && dilation[0] == 1) || \
    (dilation.size() == 2 && dilation[0] == 1 && dilation[1] == 1), dilation_err);
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
    "max_pool2d_with_indices on mlu only support input scalar type Float and Half");
  if (input.numel() == 0) return std::make_tuple(input,
    at::empty({}, input.options().dtype(at::kLong)));

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_index = cnnl_max_pool_internal(input_contiguous, kernel_size, stride, padding,
                                             ceil_mode);
  return output_index;
}

at::Tensor cnnl_avg_pool2d_backward(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    at::IntArrayRef kernel_size,
                                    at::IntArrayRef stride,
                                    at::IntArrayRef padding, bool ceil_mode,
                                    bool count_include_pad,
                                    c10::optional<int64_t> divisor_override) {
  TORCH_MLU_CHECK(self.ndimension() == 4,
                  "cnnl pool2d only support 4D input tensor currently.");
  TORCH_MLU_CHECK(at::isFloatingType(self.scalar_type()),
    "avg_pool2d_backward on mlu only support self scalar type Float and Half");
  TORCH_MLU_CHECK(at::isFloatingType(grad_output.scalar_type()),
    "avg_pool2d_backward on mlu only support grad_output scalar type Float and Half");
  if (self.numel() == 0 || grad_output.numel() == 0) return self;

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);

  return cnnl_avg_pool2d_backward_internal(grad_output_contiguous, self_contiguous,
                                           {}, kernel_size,
                                           stride, padding, ceil_mode,
                                           count_include_pad);
}

at::Tensor cnnl_max_pool2d_backward(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    at::IntArrayRef kernel_size,
                                    at::IntArrayRef stride,
                                    at::IntArrayRef padding,
                                    at::IntArrayRef dilation, bool ceil_mode,
                                    const at::Tensor& indices) {
  TORCH_MLU_CHECK(self.ndimension() == 4,
                  "cnnl pool2d only support 4D input tensor currently.");
  bool has_dilation = (dilation[0] > 1 || (dilation.size() > 1 && dilation[1] > 1));
  TORCH_MLU_CHECK(!has_dilation,
                  "cnnl pool2d dilation does not support greater than one.");
  TORCH_MLU_CHECK(at::isFloatingType(grad_output.scalar_type()),
    "max_pool2d_backward on mlu only support grad_output scalar type Float and Half");
  TORCH_MLU_CHECK(at::isFloatingType(self.scalar_type()),
    "max_pool2d_backward on mlu only support self scalar type Float and Half");
  if (self.numel() == 0 || grad_output.numel() == 0) return self;
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  return cnnl_avg_pool2d_backward_internal(
      grad_output_contiguous, self_contiguous,
      indices_contiguous, kernel_size,
      stride, padding, ceil_mode, 0);
}

at::Tensor cnnl_max_pool2d(const at::Tensor& input, at::IntArrayRef kernel_size,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
    "max_pool2d on mlu only support input scalar type Float and Half");
  if (input.numel() == 0) return input;
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  return cnnl_pool2d_internal(input_contiguous, kernel_size, stride, padding,
                              ceil_mode, false, 1);
}

at::Tensor cnnl_avg_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                           at::IntArrayRef stride, at::IntArrayRef padding,
                           bool ceil_mode, bool count_include_pad,
                           c10::optional<int64_t> divisor_override) {
  // pytorch official check
  if (self.numel() == 0) return self;
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const int64_t nslices = self.size(-4);
  const int64_t itime = self.size(-3);
  const int64_t iheight = self.size(-2);
  const int64_t iwidth = self.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
    self,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    padT, padH, padW,
    1, 1, 1,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    /*check_input_size=*/ true);

  // cnnl support datatype check
  TORCH_MLU_CHECK(Pool3dSupportDtype.find(self.scalar_type()) != Pool3dSupportDtype.end(),
                  "avg_pool3d not implemented for '", self.scalar_type(), "'");
  TORCH_MLU_CHECK(self.ndimension() == 5,
                  "cnnl avg_pool3d only support 5D input tensor currently.");
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);

  return cnnl_pool3d_internal(
      self_contiguous, kernel_size, stride, padding, ceil_mode, count_include_pad, 0);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool3d_with_indices(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  // cnnl support datatype check
  if (input.numel() == 0) return std::make_tuple(input,
    at::empty({}, input.options().dtype(at::kLong)));
  TORCH_MLU_CHECK(Pool3dSupportDtype.find(input.scalar_type()) != Pool3dSupportDtype.end(),
                  "max_pool3d_with_indices not implemented for '",
                  input.scalar_type(), "'");
  TORCH_MLU_CHECK(input.ndimension() == 5,
                  "cnnl max_pool3d only support 5D input tensor currently.");
  bool has_dilation = false;
  for (auto& d : dilation) {
    if (d > 1) {
      has_dilation = true;
      break;
    }
  }
  TORCH_MLU_CHECK(!has_dilation,
                  "cnnl max_pool3d dilation does not support greater than one.");

  auto index = at::empty({}, input.options().dtype(at::kLong));
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  return cnnl_maxpool3d_with_index_internal(input_contiguous, kernel_size, stride, padding,
                                            dilation, ceil_mode);
}

at::Tensor cnnl_avg_pool3d_backward(const at::Tensor & grad_output,
    const at::Tensor & self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding,
    bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  // cnnl support datatype check
  if (grad_output.numel() == 0 || self.numel() == 0) return self;
  TORCH_MLU_CHECK(Pool3dSupportDtype.find(self.scalar_type()) != Pool3dSupportDtype.end(),
                  "avg_pool3d_backward not implemented for '",
                  self.scalar_type(), "'");
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  return cnnl_avgpool3d_backward_internal(grad_output_contiguous, self_contiguous,
                                          kernel_size, stride, padding, ceil_mode,
                                          count_include_pad, divisor_override);
}

at::Tensor cnnl_max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  if (grad_output.numel() == 0 || self.numel() == 0) return self;
  TORCH_MLU_CHECK(Pool3dSupportDtype.find(self.scalar_type()) != Pool3dSupportDtype.end(),
                  "max_pool3d_with_indices_backward not implemented for '",
                  self.scalar_type(), "'");
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  return cnnl_maxpool3d_backward_internal(grad_output_contiguous, self_contiguous, kernel_size,
                                          stride, padding, dilation, ceil_mode, indices);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
