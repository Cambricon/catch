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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "ATen/native/UpSample.h"
using at::native::upsample_2d_shape_check;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_upsample_bilinear2d(const at::Tensor& self,
                                    at::IntArrayRef output_size,
                                    bool align_corners,
                                    c10::optional<double> scales_h,
                                    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);
  bool align_center = !align_corners;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_BILINEAR;
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output = at::empty({nbatch, channels, output_height, output_width},
                          self.options(), memory_format);
  return cnnl_upsample_internal(output, self_contiguous, output_size,
                                align_corners, align_center, interp_mode);
}

at::Tensor& cnnl_upsample_bilinear2d_out(at::Tensor& output,
                                         const at::Tensor& self,
                                         at::IntArrayRef output_size,
                                         bool align_corners,
                                         c10::optional<double> scales_h,
                                         c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output_t = at::empty({nbatch, channels, output_height, output_width},
                            self.options(), memory_format);
  bool align_center = !align_corners;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_BILINEAR;
  auto out_tmp = cnnl_upsample_internal(output_t, self_contiguous, output_size,
                                        align_corners, align_center, interp_mode);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(output_t));
  resize_impl_mlu_(getMluTensorImpl(output), output_t.sizes(),
                   output_t.strides());
  return output;
}

at::Tensor cnnl_upsample_bilinear2d_backward(const at::Tensor& grad_output,
                                             at::IntArrayRef output_size,
                                             at::IntArrayRef input_size,
                                             bool align_corners,
                                             c10::optional<double> scales_h,
                                             c10::optional<double> scales_w) {
 TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  upsample_2d_shape_check(
      at::Tensor(),
      grad_output,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto memory_format = get_channels_last_memory_format(grad_output.dim());
  auto grad_output_ = cnnl_contiguous(grad_output, memory_format);
  auto grad_input = at::empty({nbatch, channels, input_height, input_width},
                              grad_output.options(), memory_format);

  cnnlInterpBackwardMode_t interp_mode = CNNL_INTERP_BACKWARD_BILINEAR;
  bool align_center = !align_corners;
  return cnnl_upsample_backward_internal(grad_input, grad_output_, output_size, input_size,
                                align_corners, align_center, interp_mode);
}

at::Tensor cnnl_upsample_nearest2d(const at::Tensor& self,
                                   at::IntArrayRef output_size,
                                   c10::optional<double> scales_h,
                                   c10::optional<double> scales_w) {
  auto input_arg = at::TensorArg(self, "input", 1);
  at::checkScalarTypes("upsample_neareat2d", input_arg,
                       {at::ScalarType::Float, at::ScalarType::Half});
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);

  bool align_corners = false;
  bool align_center = false;

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output = at::empty({nbatch, channels, output_height, output_width},
                          self.options(), memory_format);
  cnnlInterpMode_t interp_mode = CNNL_INTERP_NEAREST;
  return cnnl_upsample_internal(output, self_contiguous, output_size,
                                align_corners, align_center, interp_mode);
}

at::Tensor& cnnl_upsample_nearest2d_out(at::Tensor& output,
                                        const at::Tensor& self,
                                        at::IntArrayRef output_size,
                                        c10::optional<double> scales_h,
                                        c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);

  bool align_corners = false;
  bool align_center = false;

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto output_t = at::empty({nbatch, channels, output_height, output_width},
                            self.options(), memory_format);
  cnnlInterpMode_t interp_mode = CNNL_INTERP_NEAREST;
  auto out_tmp = cnnl_upsample_internal(output_t, self_contiguous, output_size,
                                        align_corners, align_center, interp_mode);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(output_t));
  resize_impl_mlu_(getMluTensorImpl(output), output_t.sizes(),
                   output_t.strides());
  return output;
}

at::Tensor cnnl_upsample_nearest2d_backward(const at::Tensor& grad_output,
                                             at::IntArrayRef output_size,
                                             at::IntArrayRef input_size,
                                             c10::optional<double> scales_h,
                                             c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  upsample_2d_shape_check(
      at::Tensor(),
      grad_output,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto memory_format = get_channels_last_memory_format(grad_output.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grad_input = at::empty({nbatch, channels, input_height, input_width},
                              grad_output.options(), memory_format);

  cnnlInterpBackwardMode_t interp_mode = CNNL_INTERP_BACKWARD_NEAREST;
  bool align_center = false;
  bool align_corners = false;
  return cnnl_upsample_backward_internal(grad_input, grad_output_contiguous, output_size,
                                         input_size, align_corners, align_center, interp_mode);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
