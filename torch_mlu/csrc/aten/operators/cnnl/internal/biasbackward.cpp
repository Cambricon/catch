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
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_bias_backward_internal(const at::Tensor& input, int64_t dim) {
  auto input_shape = input.sizes().vec();
  std::vector<int64_t> out_shape = {input_shape[dim]};
  at::Tensor output = at::empty(out_shape, input.options());
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  const int input_dim = input.dim();
  // this cnnl kernel is using for conv and linear.
  // cnnl conv input tensor need be channel last or channel last 3d.
  // linear op is using matmul to caculate output, and dims == 2 always.
  // so set memory format is channel last or channel last 3d when dim is 4 or 5.
  if (input_dim == 4 || input_dim == 5) {
    memory_format = get_channels_last_memory_format(input_dim);
    dim = modify_dim_based_on_layout(dim, memory_format);
    layout = input.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  }
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto input_impl = getMluTensorImpl(input_contiguous);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  desc_input.set(input_contiguous, layout);
  desc_output.set(output);

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // compute
  TORCH_CNNL_CHECK(cnnlBiasAddBackward(handle, desc_input.desc(), input_ptr,
                                       dim, desc_output.desc(), output_ptr));
  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
