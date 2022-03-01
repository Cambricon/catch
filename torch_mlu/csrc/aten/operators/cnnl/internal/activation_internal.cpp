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

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/util/dispatch.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_activation_internal(at::Tensor& output,
                                     const at::Tensor& input,
                                     cnnlActivationMode_t mode,
                                     at::Scalar scalar) {
  // prepare cnnl input
  auto scalar_value = scalar.to<float>();
  auto input_dtype = input.scalar_type();
  TORCH_MLU_CHECK(input_dtype == at::kFloat || input_dtype == at::kHalf,
                  "input must be float/half");
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  CnnlTensorDescriptor input_desc;
  input_desc.set(input);
  CnnlActivationDescriptor op_desc;
  op_desc.set(mode, CNNL_NOT_PROPAGATE_NAN, scalar_value);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // call cnnl activation interface
  auto handle = getCurrentHandle();
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlActivationForward(
      /* handle          */ handle,
      /* activation_desc */ op_desc.desc(),
      /* alpha           */ alpha,
      /* x_desc          */ input_desc.desc(),
      /* x               */ input_ptr,
      /* beta            */ beta,
      /* y_desc          */ output_desc.desc(),
      /* y               */ output_ptr));
  return output;
}

at::Tensor cnnl_activation_backward_internal(const at::Tensor& self,
                                             const at::Tensor& grad,
                                             cnnlActivationMode_t mode) {
  // prepare cnnl input
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_ptr = grad_impl->cnnlMalloc();
  CnnlTensorDescriptor grad_desc;
  grad_desc.set(grad);
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  CnnlActivationDescriptor op_desc;
  op_desc.set(mode, CNNL_NOT_PROPAGATE_NAN, 0.0);

  // prepare cnnl output
  auto memory_format = self.suggest_memory_format();
  auto output = at::empty(grad.sizes(), grad.options(), memory_format);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // call cnnl activation interface
  auto handle = getCurrentHandle();
  const void * alpha = nullptr;
  const void * beta = nullptr;
  if (mode == CNNL_ACTIVATION_GELU) {
    TORCH_CNNL_CHECK(cnnlActivationBackward(
        /* handle          */ handle,
        /* activation_desc */ op_desc.desc(),
        /* alpha           */ alpha,
        /* y_desc          */ nullptr,
        /* y               */ nullptr,
        /* diff_y_desc     */ grad_desc.desc(),
        /* diff_y          */ grad_ptr,
        /* x_desc          */ self_desc.desc(),
        /* x               */ self_ptr,
        /* beta            */ beta,
        /* diff_x_desc     */ output_desc.desc(),
        /* diff_x          */ output_ptr));
    return output;
  }
  TORCH_CNNL_CHECK(cnnlActivationBackward(
      /* handle          */ handle,
      /* activation_desc */ op_desc.desc(),
      /* alpha           */ alpha,
      /* y_desc          */ self_desc.desc(),
      /* y               */ self_ptr,
      /* diff_y_desc     */ grad_desc.desc(),
      /* diff_y          */ grad_ptr,
      /* x_desc          */ nullptr,
      /* x               */ nullptr,
      /* beta            */ beta,
      /* diff_x_desc     */ output_desc.desc(),
      /* diff_x          */ output_ptr));
  return output;
}

at::Tensor cnnl_leaky_relu_backward_internal(const at::Tensor& self,
                                             const at::Tensor& grad,
                                             cnnlActivationMode_t mode,
                                             at::Scalar scalar,
                                             bool self_is_result) {
  auto scalar_value = scalar.to<float>();
  TORCH_CHECK(!self_is_result || scalar_value >= 0.0,
    "In-place leakyReLu backward calculation is triggered"
    " with a negative slope which is not supported. "
    "This is caused by calling in-place forward function with a negative slope, "
    "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch"
    " if you do require supporting in-place leakRelu backward calculation with negative slope");
  // prepare cnnl input
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_ptr = grad_impl->cnnlMalloc();
  CnnlTensorDescriptor grad_desc;
  grad_desc.set(grad);
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  CnnlActivationDescriptor op_desc;
  op_desc.set(mode, CNNL_NOT_PROPAGATE_NAN, scalar_value);

  // prepare cnnl output
  auto memory_format = self.suggest_memory_format();
  auto output = at::empty(grad.sizes(), grad.options(), memory_format);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // call cnnl activation interface
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlActivationBackward(
      /* handle          */ handle,
      /* activation_desc */ op_desc.desc(),
      /* alpha           */ nullptr,
      /* y_desc          */ nullptr,
      /* y               */ nullptr,
      /* diff_y_desc     */ grad_desc.desc(),
      /* diff_y          */ grad_ptr,
      /* x_desc          */ self_desc.desc(),
      /* x               */ self_ptr,
      /* beta            */ nullptr,
      /* diff_x_desc     */ output_desc.desc(),
      /* diff_x          */ output_ptr));
  return output;
}

at::Tensor cnnl_hardtanh_backward_internal(const at::Tensor& self,
                                           const at::Tensor& grad,
                                           at::Scalar min_val,
                                           at::Scalar max_val) {
  auto memory_format = self.suggest_memory_format();
  auto output = at::empty(self.sizes(), self.options(), memory_format);
  const float min_f = min_val.to<float>();
  const float max_f = max_val.to<float>();
  // prepare cnnl input
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_ptr = grad_impl->cnnlMalloc();
  CnnlTensorDescriptor grad_desc;
  grad_desc.set(grad);
  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlHardtanhBackward(
      /* handle      */ handle,
      /* x_desc      */ self_desc.desc(),
      /* x           */ self_ptr,
      /* diff_y_desc */ grad_desc.desc(),
      /* diff_y      */ grad_ptr,
      /* max_val     */ max_f,
      /* min_val     */ min_f,
      /* diff_x_desc */ output_desc.desc(),
      /* diff_x      */ output_ptr));
  return output;
}

at::Tensor& cnnl_hardtanh_internal(at::Tensor&output,
                                   const at::Tensor& input,
                                   at::Scalar min_val,
                                   at::Scalar max_val) {
  const float min_f = min_val.to<float>();
  const float max_f = max_val.to<float>();
  // prepare cnnl input
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  CnnlTensorDescriptor input_desc;
  input_desc.set(input);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlHardtanh(
      /* handle  */ handle,
      /* x_desc  */ input_desc.desc(),
      /* x       */ input_ptr,
      /* max_val */ max_f,
      /* min_val */ min_f,
      /* y_desc  */ output_desc.desc(),
      /* y       */ output_ptr));
  return output;
}

at::Tensor& cnnl_softplus_internal(at::Tensor&output,
                                   const at::Tensor& input,
                                   at::Scalar beta,
                                   at::Scalar threshold) {
  const int beta_t = beta.to<int>();
  const int threshold_t = threshold.to<int>();
  // prepare cnnl input
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  CnnlTensorDescriptor input_desc;
  input_desc.set(input);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlSoftplusForward(
                         /* handle  */ handle,
                         /* x_desc  */ input_desc.desc(),
                         /* x       */ input_ptr,
                         /* y_desc  */ output_desc.desc(),
                         /* y       */ output_ptr,
                         /* beta    */ beta_t,
                       /* threshold */ threshold_t));
  return output;
}

at::Tensor& cnnl_softplus_backward_internal(at::Tensor& grad_input,
                                           const at::Tensor& self,
                                           const at::Tensor& grad_output,
                                           at::Scalar beta,
                                           at::Scalar threshold,
                                           const at::Tensor& output) {
  const int beta_t = beta.to<int>();
  const int threshold_t = threshold.to<int>();
  // prepare cnnl self
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->cnnlMalloc();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  // prepare cnnl grad_input
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  CnnlTensorDescriptor grad_input_desc;
  grad_input_desc.set(grad_input);
  auto handle = getCurrentHandle();
  // prepare cnnl grad_output
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  CnnlTensorDescriptor grad_output_desc;
  grad_output_desc.set(grad_output);
  TORCH_CNNL_CHECK(cnnlSoftplusBackward(
      /* handle      */ handle,
      /* x_desc      */ grad_output_desc.desc(),
      /* x           */ grad_output_ptr,
      /* diff_y_desc */ self_desc.desc(),
      /* diff_y      */ self_ptr,
      /* diff_x_desc */ grad_input_desc.desc(),
      /* diff_x      */ grad_input_ptr,
      /* beta        */ beta_t,
      /* threshold   */ threshold_t));
  return grad_input;
}

at::Tensor& cnnl_threshold_internal(at::Tensor& output, const at::Tensor& input,
                                    at::Scalar threshold, at::Scalar value) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input);
  output_desc.set(output);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();

  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF(input.scalar_type(), "cnnl_threshold", [&] {
    auto threshold_val = threshold.to<scalar_t>();
    auto value_val = value.to<scalar_t>();
    TORCH_CNNL_CHECK(cnnlThreshold(handle, input_desc.desc(), input_ptr,
      static_cast<void*>(&threshold_val), static_cast<void*>(&value_val),
      output_desc.desc(), output_ptr));
  });

  return output;
}

at::Tensor& cnnl_threshold_backward_internal(at::Tensor& grad_input, const at::Tensor& grad_output,
                                             const at::Tensor& input, at::Scalar threshold) {
  CnnlTensorDescriptor grad_input_desc;
  CnnlTensorDescriptor grad_output_desc;
  CnnlTensorDescriptor input_desc;
  grad_input_desc.set(grad_input);
  grad_output_desc.set(grad_output);
  input_desc.set(input);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto grad_input_ptr = grad_input_impl->cnnlMalloc();
  auto grad_output_ptr = grad_output_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();

  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF(input.scalar_type(), "cnnl_threshold_backward", [&] {
    auto threshold_val = threshold.to<scalar_t>();
    TORCH_CNNL_CHECK(cnnlThresholdBackward(handle, input_desc.desc(), input_ptr,
      grad_output_desc.desc(), grad_output_ptr, static_cast<void*>(&threshold_val),
      grad_input_desc.desc(), grad_input_ptr));
  });

  return grad_input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

