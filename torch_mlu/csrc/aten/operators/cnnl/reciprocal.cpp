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
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_reciprocal(const at::Tensor& input) {
    auto input_contiguous = cnnl_contiguous(input, input.suggest_memory_format());
    auto input_tensor = input_contiguous;
    if (input.dim() == 0) {
        input_tensor = convertScalarToTensor(input_contiguous);
    }
    auto output_tensor = at::empty_like(input_tensor);
    return cnnl_reciprocal_internal(output_tensor, input_tensor);
}

at::Tensor & cnnl_reciprocal_(at::Tensor& input) {
    auto input_tensor = input;
    if (input.dim() == 0) {
        input_tensor = convertScalarToTensor(input);
    }
    return cnnl_reciprocal_internal(input_tensor, input_tensor);
}

at::Tensor & cnnl_reciprocal_out(at::Tensor & out, const at::Tensor& input) {
    auto input_contiguous = cnnl_contiguous(input, input.suggest_memory_format());
    auto input_tensor = input_contiguous;
    if (input.dim() == 0) {
        input_tensor = convertScalarToTensor(input_contiguous);
    }
    auto output_tensor = at::empty_like(input_tensor);
    cnnl_reciprocal_internal(output_tensor, input_tensor);
    getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output_tensor));
    resize_impl_mlu_(getMluTensorImpl(out), output_tensor.sizes(), output_tensor.strides());
    return out;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
