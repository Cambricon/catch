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

#include <ATen/native/Activation.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

#include "aten/core/DispatchStub.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

template <typename OutImpl, typename... ArgT>
static inline Tensor dispatch_to_out_op(OutImpl& out_impl, const Tensor& self, ArgT&&... args) {
  at::Tensor result = at::empty({0}, self.options().device(at::Device::Type::MLU));
  return out_impl(result, self, std::forward<ArgT>(args)...);
}

// relu

at::Tensor cnnl_relu(const at::Tensor& input) {
  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output = at::empty(input.sizes(), input.options(), memory_format);
  return cnnl_activation_internal(output, input_contiguous, CNNL_ACTIVATION_RELU);
}

at::Tensor& cnnl_relu_(at::Tensor& input) {
  return cnnl_activation_internal(input, input, CNNL_ACTIVATION_RELU);
}

// softplus
void softplus_mlu_kernel(at::TensorIterator& iter, at::Scalar beta_, at::Scalar threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "softplus_mlu", [&]() {
    auto output = iter.output(0);
    std::vector<at::Tensor> list;
    get_contiguous(iter, list);
    auto self = list[0];
    cnnl_softplus_internal(output, self, beta_, threshold_);
  });
}

at::Tensor& cnnl_softplus_out(at::Tensor& result,
                              const at::Tensor& self,
                              at::Scalar beta = 1,
                              at::Scalar threshold = 20) {
  return at::native::softplus_out(result, self, beta, threshold);
}

at::Tensor cnnl_softplus(const at::Tensor& self,
                         at::Scalar beta = 1,
                         at::Scalar threshold = 20) {
  return dispatch_to_out_op(cnnl_softplus_out, self, beta, threshold);
}

at::Tensor cnnl_softplus_backward(const at::Tensor& grad_output,
                                  const at::Tensor& self,
                                  at::Scalar beta,
                                  at::Scalar threshold,
                                  const at::Tensor& output) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grad_input = at::empty_like(self_contiguous);
  return cnnl_softplus_backward_internal(grad_input,
                                         grad_output_contiguous,
                                         self_contiguous,
                                         beta,
                                         threshold,
                                         output);
}

// tanh
void tanh_mlu_kernel(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "tanh_mlu", [&]() {
    auto output = iter.output(0);
    std::vector<at::Tensor> list;
    get_contiguous(iter, list);
    auto self = list[0];
    cnnl_activation_internal(output, self, CNNL_ACTIVATION_TANH);
  });
}


at::Tensor& cnnl_tanh_out(at::Tensor& result, const at::Tensor& input) {
  return at::native::tanh_out(result, input);
}

at::Tensor cnnl_tanh(const at::Tensor& input) {
  return dispatch_to_out_op(cnnl_tanh_out, input);
}

at::Tensor& cnnl_tanh_(at::Tensor& input) {
  return cnnl_tanh_out(input, input);
}

at::Tensor cnnl_tanh_backward(const at::Tensor& grad, const at::Tensor& self) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_contiguous = cnnl_contiguous(grad, memory_format);
  return cnnl_activation_backward_internal(self_contiguous, grad_contiguous, CNNL_ACTIVATION_TANH);
}

// gelu

at::Tensor cnnl_gelu(const at::Tensor& input) {
  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output = at::empty(input.sizes(), input.options(), memory_format);
  return cnnl_activation_internal(output, input_contiguous, CNNL_ACTIVATION_GELU);
}

at::Tensor cnnl_gelu_backward(const at::Tensor& grad, const at::Tensor& self) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_contiguous = cnnl_contiguous(grad, memory_format);
  return cnnl_activation_backward_internal(self_contiguous, grad_contiguous, CNNL_ACTIVATION_GELU);
}

// hardtanh

at::Tensor cnnl_hardtanh(const at::Tensor& input, at::Scalar min_val, at::Scalar max_val) {
  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output = at::empty(input.sizes(), input.options(), memory_format);
  return cnnl_hardtanh_internal(output, input_contiguous, min_val, max_val);
}

at::Tensor& cnnl_hardtanh_(at::Tensor& input, at::Scalar min_val, at::Scalar max_val) {
  return cnnl_hardtanh_internal(input, input, min_val, max_val);
}

at::Tensor cnnl_hardtanh_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                  at::Scalar min_val, at::Scalar max_val) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  return cnnl_hardtanh_backward_internal(self_contiguous, grad_output_contiguous, min_val, max_val);
}

// sigmoid

at::Tensor& cnnl_sigmoid_out(Tensor& result, const Tensor& self) {
  return at::native::sigmoid_out(result, self);
}

at::Tensor cnnl_sigmoid(const at::Tensor& self) {
  return dispatch_to_out_op(cnnl_sigmoid_out, self);
}

at::Tensor& cnnl_sigmoid_(at::Tensor& self) {
  return cnnl_sigmoid_out(self, self);
}

void sigmoid_mlu_kernel(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "sigmoid_mlu", [&]() {
    auto output = iter.output(0);
    std::vector<at::Tensor> list;
    get_contiguous(iter, list);
    auto self = list[0];
    cnnl_activation_internal(output, self, CNNL_ACTIVATION_SIGMOID);
  });
}

at::Tensor cnnl_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  return cnnl_activation_backward_internal(self_contiguous,
                                           grad_output_contiguous,
                                           CNNL_ACTIVATION_SIGMOID);
}

// leaky_relu

void leaky_relu_mlu_kernel(at::TensorIterator& iter, at::Scalar negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "leaky_relu_mlu", [&]() {
    auto output = iter.output(0);
    std::vector<at::Tensor> list;
    get_contiguous(iter, list);
    auto self = list[0];
    cnnl_activation_internal(output, self, CNNL_ACTIVATION_LEAKYRELU, negval_);
  });
}

at::Tensor cnnl_leaky_relu(const at::Tensor& self, at::Scalar negative_slope) {
  return dispatch_to_out_op(cnnl_leaky_relu_out, self, negative_slope);
}

at::Tensor& cnnl_leaky_relu_(at::Tensor& input, at::Scalar negative_slope) {
  return cnnl_leaky_relu_out(input, input, negative_slope);
}

at::Tensor& cnnl_leaky_relu_out(at::Tensor & out, const at::Tensor& input,
    at::Scalar negative_slope) {
  return at::native::leaky_relu_out(out, input, negative_slope);
}

at::Tensor cnnl_leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                    at::Scalar negative_slope, bool self_is_result) {
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  return cnnl_leaky_relu_backward_internal(self_contiguous, grad_output_contiguous,
                                           CNNL_ACTIVATION_LEAKYRELU,
                                           negative_slope, self_is_result);
}

void threshold_mlu_kernel(
    at::TensorIterator& iter,
    at::Scalar threshold_scalar,
    at::Scalar value_scalar) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "threshold_mlu", [&] {
    auto output = iter.output(0);
    std::vector<at::Tensor> list;
    get_contiguous(iter, list);
    auto self = list[0];
    cnnl_threshold_internal(output, self, threshold_scalar, value_scalar);
  });
}

at::Tensor& cnnl_threshold_out(at::Tensor& out, const at::Tensor& self,
                               at::Scalar threshold, at::Scalar value) {
  TORCH_MLU_CHECK(out.scalar_type() == self.scalar_type(),
    "The datatype of out in cnnl_threshold_out must be same as self ",
    self.scalar_type(), " but out ", out.scalar_type());
  return at::native::threshold_out(out, self, threshold, value);
}

at::Tensor cnnl_threshold(const at::Tensor& self, at::Scalar threshold, at::Scalar value) {
  return dispatch_to_out_op(cnnl_threshold_out, self, threshold, value);
}

at::Tensor& cnnl_threshold_(at::Tensor& self, at::Scalar threshold, at::Scalar value) {
  cnnl_threshold_out(self, self, threshold, value);
  return self;
}

at::Tensor cnnl_threshold_backward(const at::Tensor& grad, const at::Tensor& self,
                                   at::Scalar thresold) {
  auto memory_format = self.suggest_memory_format();
  auto grad_contiguous = cnnl_contiguous(grad, memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto out = at::empty(self.sizes(), self.options(), memory_format);
  cnnl_threshold_backward_internal(out, grad_contiguous, self_contiguous, thresold);
  return out;
}

using namespace at::native;
REGISTER_MLU_DISPATCH(softplus_stub, &softplus_mlu_kernel);
REGISTER_MLU_DISPATCH(sigmoid_stub, &sigmoid_mlu_kernel);
REGISTER_MLU_DISPATCH(leaky_relu_stub, &leaky_relu_mlu_kernel);
REGISTER_MLU_DISPATCH(tanh_stub, &tanh_mlu_kernel);
REGISTER_MLU_DISPATCH(threshold_stub, &threshold_mlu_kernel);

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
