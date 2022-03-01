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

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_quantize_param(
    const at::Tensor &input, int bitwidth, cnnlQuantizeMode_t mode) {

  TORCH_MLU_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                  "quantize_param only support input float/half");

  auto position = at::empty(1, input.options().dtype(at::ScalarType::Int));

  auto input_impl = getMluTensorImpl(input);
  auto position_impl = getMluTensorImpl(position);

  CnnlTensorDescriptor input_desc;
  at::Tensor workspace, scale, offset;

  void *workspace_ptr = nullptr;
  void *scale_ptr = nullptr;
  void *offset_ptr = nullptr;

  size_t workspace_size = 0;

  // get current handle
  auto handle = getCurrentHandle();
  input_desc.set(input);

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto position_ptr = position_impl->cnnlMalloc();

  // prepare workspace
  TORCH_CNNL_CHECK(cnnlGetQuantizeParamWorkspaceSize(handle, input_desc.desc(),
                                                     &workspace_size));
  if (mode == CNNL_QUANTIZE_POSITION_SCALE) {
    scale = at::empty(1, input.options().dtype(at::ScalarType::Float));
    scale_ptr = getMluTensorImpl(scale)->cnnlMalloc();
  } else if (mode == CNNL_QUANTIZE_POSITION_SCALE_OFFSET) {
    scale = at::empty(1, input.options().dtype(at::ScalarType::Float));
    scale_ptr = getMluTensorImpl(scale)->cnnlMalloc();
    offset = at::empty(1, input.options().dtype(at::ScalarType::Float));
    offset_ptr = getMluTensorImpl(offset)->cnnlMalloc();
  }
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, input.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }

  TORCH_CNNL_CHECK(cnnlQuantizeParam(handle,
                                     mode,
                                     input_desc.desc(),
                                     input_ptr,
                                     bitwidth,
                                     workspace_ptr,
                                     workspace_size,
                                     position_ptr,
                                     scale_ptr,
                                     offset_ptr));
  return std::make_tuple(position, scale, offset);
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
