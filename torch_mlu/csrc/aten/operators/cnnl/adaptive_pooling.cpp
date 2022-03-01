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

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_adaptive_avg_pool2d_out(
  at::Tensor& output, const at::Tensor& self, at::IntArrayRef output_size) {
  TORCH_MLU_CHECK((output.scalar_type() == at::ScalarType::Float && \
                  self.scalar_type() == at::ScalarType::Float) || \
                  (output.scalar_type() == at::ScalarType::Half && \
                  self.scalar_type() == at::ScalarType::Half),
    "cnnl_adaptive_avg_pool2d_out only support half or float input currently, "
    "but output is ", output.scalar_type(), ", self is ", self.scalar_type());
  TORCH_MLU_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_MLU_CHECK(output_size.size() == 2,
    "adaptive_avg_pool2d: internal error: output_size.size() must be 2");
  for (int64_t i = 0; i < self.ndimension(); i++) {
    TORCH_MLU_CHECK(self.size(i) > 0,
    "adaptive_avg_pool2d(): expected input to have non-empty spatial dimensions, "
    "but input has sizes ", self.sizes(), " with dimension ", i, " being empty");
  }

  at::Tensor self_4d = self;
  if (self.ndimension() == 3) {
    self_4d = cnnl_unsqueeze(self, 0);
  }

  resize_impl_mlu_(getMluTensorImpl(output), {self_4d.size(0), self_4d.size(1),
    output_size[0], output_size[1]}, c10::nullopt);
  auto memory_format = get_channels_last_memory_format(self_4d.dim());
  getMluTensorImpl(output)->empty_tensor_restride(memory_format);
  auto self_contiguous = cnnl_contiguous(self_4d, memory_format);
  cnnl_adaptive_avg_pool_internal(output, self_contiguous, output_size);

  if (self.ndimension() == 3) {
    return output.squeeze_(0);
  }

  return output;
}

at::Tensor cnnl__adaptive_avg_pool2d(
  const at::Tensor& self, at::IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  cnnl_adaptive_avg_pool2d_out(output, self, output_size);
  return output;
}

at::Tensor cnnl__adaptive_avg_pool2d_backward(
  const at::Tensor& grad_output, const at::Tensor& input) {
  at::Tensor input_4d = input, grad_output_4d = grad_output;
  if (input.ndimension() == 3) {
    input_4d = at::unsqueeze(input, 0);
    grad_output_4d = at::unsqueeze(grad_output, 0);
  }
  auto memory_format = get_channels_last_memory_format(input_4d.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output_4d, memory_format);
  auto input_contiguous = cnnl_contiguous(input_4d, memory_format);

  auto grad_input = at::empty_like(input_contiguous);
  cnnl_adaptive_avg_pool_backward_internal(grad_input,
                                           grad_output_contiguous,
                                           input_contiguous);

  if (input.ndimension() == 3) {
    return grad_input.squeeze_(0);
  }

  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> cnnl_adaptive_max_pool2d(
  const at::Tensor & self, at::IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  auto indices = at::empty({0}, self.options().dtype(at::kLong));
  cnnl_adaptive_max_pool2d_out(output, indices, self, output_size);
  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_adaptive_max_pool2d_out(
  at::Tensor & output, at::Tensor & indices,
  const at::Tensor & self, at::IntArrayRef output_size) {
  TORCH_MLU_CHECK((output.scalar_type() == at::ScalarType::Float && \
                  self.scalar_type() == at::ScalarType::Float) || \
                  (output.scalar_type() == at::ScalarType::Half && \
                  self.scalar_type() == at::ScalarType::Half),
    "cnnl_adaptive_max_pool2d_out only support half or float input currently, "
    "but output is ", output.scalar_type(), ", input is ", self.scalar_type());
  TORCH_MLU_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_MLU_CHECK(output_size.size() == 2,
    "adaptive_max_pool2d: internal error: output_size.size() must be 2");
  for (int64_t i = 0; i < self.ndimension(); i++) {
    TORCH_MLU_CHECK(self.size(i) > 0,
    "adaptive_max_pool2d(): expected input to have non-empty spatial dimensions, "
    "but input has sizes ", self.sizes(), " with dimension ", i, " being empty");
  }

  auto self_4d = self;
  if (self.ndimension() == 3) {
    self_4d = at::unsqueeze(self, 0);
  }

  // TODO(zhanchendi):
  // cnnl only accept these dtypes half-half-int16 or float-float-int32（input-output-indices）
  auto indices_mlu = indices;
  if (self.scalar_type() == at::ScalarType::Half) {
    indices_mlu = indices.to(at::kShort, true);
  }

  resize_impl_mlu_(getMluTensorImpl(output), {self_4d.size(0), self_4d.size(1),
    output_size[0], output_size[1]}, c10::nullopt);
  resize_impl_mlu_(getMluTensorImpl(indices_mlu), {self_4d.size(0), self_4d.size(1),
    output_size[0], output_size[1]}, c10::nullopt);
  auto memory_format = get_channels_last_memory_format(self_4d.dim());
  getMluTensorImpl(output)->empty_tensor_restride(memory_format);
  getMluTensorImpl(indices_mlu)->empty_tensor_restride(memory_format);
  auto self_contiguous = cnnl_contiguous(self_4d, memory_format);
  cnnl_adaptive_max_pool2d_internal(output, indices_mlu, self_contiguous, output_size);

  if (self.ndimension() == 3) {
    output.squeeze_(0);
    indices_mlu.squeeze_(0);
  }

  if (self.scalar_type() == at::ScalarType::Half) {
    resize_impl_mlu_(getMluTensorImpl(indices), indices_mlu.sizes(), indices_mlu.strides());
    indices.copy_(indices_mlu, true);
  }

  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

at::Tensor cnnl_adaptive_max_pool2d_backward(
  const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & indices) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  cnnl_adaptive_max_pool2d_backward_out(grad_input, grad_output, input, indices);
  return grad_input;
}

at::Tensor& cnnl_adaptive_max_pool2d_backward_out(
  at::Tensor& grad_input, const at::Tensor & grad_output,
  const at::Tensor & input, const at::Tensor & indices) {
  at::Tensor input_4d = input, grad_output_4d = grad_output, indices_4d = indices;
  if (input.ndimension() == 3) {
    input_4d = at::unsqueeze(input, 0);
    grad_output_4d = at::unsqueeze(grad_output, 0);
    indices_4d = at::unsqueeze(indices, 0);
  }

  // TODO(zhanchendi):
  // cnnl only accept these dtypes half-half-int16 or float-float-int32（input-output-indices）
  auto indices_4d_mlu = indices_4d;
  if (input.scalar_type() == at::ScalarType::Half) {
    indices_4d_mlu = indices_4d.to(at::kShort, true);
  }

  resize_impl_mlu_(getMluTensorImpl(grad_input), input_4d.sizes(), c10::nullopt);
  auto memory_format = get_channels_last_memory_format(input_4d.dim());
  getMluTensorImpl(grad_input)->empty_tensor_restride(memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output_4d, memory_format);
  auto input_contiguous = cnnl_contiguous(input_4d, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices_4d_mlu, memory_format);

  cnnl_adaptive_max_pool2d_backward_internal(grad_input,
                                             grad_output_contiguous,
                                             input_contiguous,
                                             indices_contiguous);

  if (input.ndimension() == 3) {
    return grad_input.squeeze_(0);
  }

  return grad_input;
}

at::Tensor cnnl_adaptive_avg_pool3d(const at::Tensor & input, at::IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(input.ndimension() == 5, "non-empty 5D tensor expected for input");

  /* size */
  int64_t sizeD = input.size(-4);
  /* output sizes */
  auto osizeT = output_size[0];
  auto osizeH = output_size[1];
  auto osizeW = output_size[2];
  std::vector<int64_t> output_shape = {
      input.size(-5), sizeD, osizeT, osizeH, osizeW};
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output = at::empty(output_shape, input.options());
  getMluTensorImpl(output)->empty_tensor_restride(memory_format);
  cnnl_adaptive_avg_pool_internal(output, input_contiguous, output_size);
  return output;
}

at::Tensor cnnl_adaptive_avg_pool3d_backward(const at::Tensor & grad_output,
                                             const at::Tensor & self) {
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);

  auto grad_input = at::empty_like(self_contiguous);
  cnnl_adaptive_avg_pool_backward_internal(grad_input, grad_output_contiguous, self_contiguous);
  return grad_input;
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
