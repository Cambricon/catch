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

at::Tensor cnnl_gather_internal(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index) {
  auto self_contiguous = cnnl_contiguous(self);
  auto index_contiguous = cnnl_contiguous(index);
  int64_t ndim = at::native::ensure_nonempty_dim(self_contiguous.dim());
  dim = at::maybe_wrap_dim(dim, self_contiguous);
  auto index_contiguous_type = index_contiguous.scalar_type();
  TORCH_MLU_CHECK(index_contiguous_type == at::kInt || index_contiguous_type == at::kLong,
                  "index dtype should be int/long");
  TORCH_MLU_CHECK(ndim == index_contiguous.dim(), "self and index must have same dim");
  TORCH_MLU_CHECK(ndim > dim, "self.dim() must not be less than dim");
  auto output = at::empty_like(index_contiguous, self_contiguous.options());
  for (int i = 0; i < ndim; i++) {
    TORCH_MLU_CHECK(index_contiguous.sizes()[i] == output.sizes()[i], "index and output ", i,
                    " have unsame dim");
    if (i != dim) {
      TORCH_MLU_CHECK(self_contiguous.sizes()[i] == output.sizes()[i], "self and output ", i,
                    " have unsame dim");
    }
  }
  auto input_impl = getMluTensorImpl(self_contiguous);
  auto indices_impl = getMluTensorImpl(index_contiguous);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor indices_desc;
  CnnlTensorDescriptor output_desc;
  if (self.dim() != index.dim()) {
    input_desc.set_dim(self_contiguous);
    indices_desc.set_dim(index_contiguous);
    output_desc.set_dim(output);
  } else {
    input_desc.set_dim(self_contiguous, 4);
    indices_desc.set_dim(index_contiguous, 4);
    output_desc.set_dim(output, 4);
  }
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto indices_ptr = indices_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlGather(handle, dim, input_desc.desc(), input_ptr,
                              indices_desc.desc(), indices_ptr,
                              output_desc.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
