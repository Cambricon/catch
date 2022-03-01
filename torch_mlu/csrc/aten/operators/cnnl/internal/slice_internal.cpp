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

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/Storage.h"
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_slice_internal(const at::Tensor& input, int64_t dim,
                               int64_t start, int64_t end, int64_t step) {
  auto ndim = input.dim();
  TORCH_MLU_CHECK(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  TORCH_MLU_CHECK(step > 0, "slice step must be positive");
  auto memory_format = input.suggest_memory_format();
  TORCH_MLU_CHECK(input.is_contiguous(memory_format), "Input tensor need be contiguous.");
  dim = ::at::maybe_wrap_dim(dim, ndim);
  auto sizes = input.sizes().vec();
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }

  // currently we support partial storage sharing
  sizes[dim] = (end - start + step - 1) / step;
  if (input.dim() == 1 && dim == 0 && step == 1) {
    auto inplace_output = at::detail::make_tensor<MLUTensorImpl>(
        c10::Storage(input.storage()), input.key_set(), input.dtype());
    auto strides = input.strides().vec();
    at::native::setStrided(inplace_output, sizes, strides,
                           input.storage_offset() + start * strides[dim]);
    return inplace_output;
  }

  if (step == 1 && end - start == input.size(dim)) return input;

  at::Tensor output = at::empty(sizes, input.options(), memory_format);
  if (output.numel() == 0) {
      return output;
  }

  // Modify size info for create desc.
  dim = modify_dim_based_on_layout(dim, memory_format);
  auto input_size = modify_dims_based_on_layout(input.sizes().vec(), memory_format);
  auto output_size = modify_dims_based_on_layout(output.sizes().vec(), memory_format);

  // Modify cnnl start/end/step info based on real layout.
  std::vector<int> starts(ndim, 0);
  std::vector<int> ends(ndim, 0);
  std::vector<int> steps(ndim, 1);

  for (int i = 0; i < ndim; ++i) {
    ends[i] = static_cast<int>(input_size[i]);
  }

  starts[dim] = static_cast<int>(start);
  ends[dim] = static_cast<int>(end);
  steps[dim] = static_cast<int>(step);

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input, input_size,
                 get_contiguous_strides(input_size),
                 CNNL_LAYOUT_ARRAY);
  output_desc.set(output, output_size,
                 get_contiguous_strides(output_size),
                 CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlStridedSlice(
      handle, input_desc.desc(), input_ptr, starts.data(),
      ends.data(), steps.data(), output_desc.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
