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

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_pow_internal(at::Tensor& output, const at::Tensor& input,
                               const at::Tensor& exponent) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descExp;
  CnnlTensorDescriptor descOutput;

  auto cnnl_layout = suggest_cnnl_layout(input);

  descInput.set(input, cnnl_layout);
  descExp.set(exponent, cnnl_layout);
  descOutput.set(output, cnnl_layout);
  size_t sz;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetPowWorkspaceSize(handle, descInput.desc(),
    descExp.desc(), descOutput.desc(), &sz));

  auto workspace = at::empty(sz, input.options().dtype(at::ScalarType::Char));
  auto input_impl = getMluTensorImpl(input);
  auto exp_impl = getMluTensorImpl(exponent);
  auto output_impl = getMluTensorImpl(output);
  auto workspace_impl = getMluTensorImpl(workspace);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto exp_ptr = exp_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  // set descriptor config
  cnnlComputationPreference_t high_precision = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlPow(handle, high_precision, descInput.desc(), input_ptr, descExp.desc(),
    exp_ptr, workspace_ptr, sz, descOutput.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
