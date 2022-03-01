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

at::Tensor cnnl_pow(const at::Tensor& input, at::Scalar exponent) {
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
                  "cnnl_pow only support Float and Half data types, but got data type",
                  input.scalar_type());
  auto input_contiguous = cnnl_contiguous(input, input.suggest_memory_format());
  auto output = at::empty_like(input_contiguous);
  at::Tensor tensor_exp = at::full({1}, exponent, input_contiguous.options());
  return cnnl_pow_internal(output, input_contiguous, tensor_exp);
}

at::Tensor cnnl_pow(at::Scalar input, const at::Tensor& exponent) {
  TORCH_MLU_CHECK(at::isFloatingType(exponent.scalar_type()),
                  "cnnl_pow only support Float and Half data types, but got data type",
                  exponent.scalar_type());
  auto exponent_contiguous = cnnl_contiguous(exponent, exponent.suggest_memory_format());
  auto tensor_input = at::full_like(exponent_contiguous, input,
                                    exponent_contiguous.options());
  auto input_contiguous = cnnl_contiguous(tensor_input, exponent.suggest_memory_format());
  auto output = at::empty_like(input_contiguous);
  return cnnl_pow_internal(output, input_contiguous, exponent_contiguous);
}

at::Tensor cnnl_pow(const at::Tensor& input, const at::Tensor& exponent) {
  TORCH_MLU_CHECK(input.scalar_type() == exponent.scalar_type(),
                  "cnnl_pow expected input.dtype == exponent.dtype, but got input.dtype ",
                  input.scalar_type(), " and exponent.dtype ", exponent.scalar_type());
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
                  "cnnl_pow only support Float and Half data types, but got data type",
                  input.scalar_type());
  auto output_shape = at::infer_size(input.sizes(), exponent.sizes());
  auto input_memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, input_memory_format);
  auto exponent_contiguous = cnnl_contiguous(exponent, input_memory_format);
  auto output = at::empty(output_shape, input.options(),
                          input_memory_format);
  return cnnl_pow_internal(output, input_contiguous, exponent_contiguous);
}

at::Tensor& cnnl_pow_(at::Tensor& input, at::Scalar exponent) {
  TORCH_MLU_CHECK(at::isFloatingType(input.scalar_type()),
                  "cnnl_pow only support Float and Half data types, but got data type",
                  input.scalar_type());
  at::Tensor tensor_exp = at::full({1}, exponent, input.options());
  return cnnl_pow_internal(input, input, tensor_exp);
}

at::Tensor& cnnl_pow_(at::Tensor& input, const at::Tensor& exponent) {
  // cnnlPow does not support inplace not dense
  auto tmp = cnnl_pow(input, exponent);
  input.copy_(tmp);
  return input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
