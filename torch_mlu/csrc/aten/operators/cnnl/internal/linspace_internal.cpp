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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

template <typename T>
void linspace_kernel(at::Scalar start, at::Scalar end, int64_t steps, at::Tensor& output) {
  auto start_val = start.to<T>();
  auto end_val = end.to<T>();
  auto steps_val = (end_val - start_val) / (steps - 1);
  auto end_change = end_val + steps_val;
  auto* output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
  auto handle = getCurrentHandle();
  auto output_type = output_impl->getCnnlType();
  TORCH_CNNL_CHECK(cnnlArange(handle, static_cast<void*>(&start_val),
                              static_cast<void*>(&end_change),
                              static_cast<void*>(&steps_val),
                              output_type,
                              output_ptr));
}

at::Tensor cnnl_linspace_internal(at::Tensor& output, at::Scalar start,
    at::Scalar end, int64_t steps) {
  if (output.numel() != steps) {
    output.resize_({steps});
  }

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    output.fill_(start);
  } else {
    if (at::isFloatingType(output.scalar_type())) {
      linspace_kernel<float>(start, end, steps, output);
    } else if (at::isIntegralType(output.scalar_type())) {
      linspace_kernel<int>(start, end, steps, output);
    } else {
      auto output_float = at::empty(output.sizes(), output.options().dtype(at::kFloat));
      linspace_kernel<float>(start, end, steps, output);
      cnnl_cast_internal(output_float, output);
    }
  }
  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

