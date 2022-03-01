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

at::Tensor& cnnl_masked_fill_(at::Tensor& input, const at::Tensor& mask,
                              const at::Tensor& value) {
  auto input_arg = at::TensorArg(input, "input", 1);
  at::checkScalarTypes("masked_fill", input_arg, {at::ScalarType::Float, at::ScalarType::Int,
                       at::ScalarType::Half, at::ScalarType::Double});
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Char ||
              mask.scalar_type() == at::ScalarType::Bool,
              "mask tensor only support uint8 or bool, but got ", mask.scalar_type());
  TORCH_CHECK(value.dim() == 0,
              "masked_fill_ only supports a 0-dimensional value tensor, but got tensor ",
              "with ", value.dim(), " dimension(s).");
  TORCH_CHECK(value.scalar_type() != at::ScalarType::Byte, "value tensor does not suppot uint8");
  if (mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, " \
               "this behavior is now deprecated," \
               "please use a mask with dtype torch.bool instead.");
  }
  return cnnl_masked_fill_internal(input, input, mask, value);
}

at::Tensor& cnnl_masked_fill_(at::Tensor& input, const at::Tensor& mask,
                              at::Scalar value) {
  auto value_tensor = at::full({}, value.to<float>(), input.options());
  return cnnl_masked_fill_(input, mask, value_tensor);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
