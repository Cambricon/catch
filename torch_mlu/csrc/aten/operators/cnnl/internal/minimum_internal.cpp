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

#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_minimum_internal(const at::Tensor& input, const at::Tensor& other) {
  // If the input tensor can be broadcasted, need to fill the dimension.
  // Otherwise, the conversion of layout will change to not comply with
  // the broadcasting rules.
  auto tensor_vec = broadcast_tensor(input, other, input.scalar_type());
  auto input_tensor = std::get<0>(tensor_vec);
  auto other_tensor = std::get<1>(tensor_vec);
  // get the shapes of tensors
  auto input_shape = input_tensor.sizes();
  auto other_shape = other_tensor.sizes();
  auto output_shape = at::infer_size(input_shape, other_shape);

  reshapeTo(input_tensor, input);
  reshapeTo(other_tensor, other);

  auto memory_format = input_tensor.suggest_memory_format();
  auto output = at::empty(output_shape, input_tensor.options(), memory_format);
  if (output.numel() == 0)
    return output;
  auto input_contiguous = cnnl_contiguous(input_tensor, memory_format);
  auto other_contiguous = cnnl_contiguous(other_tensor, memory_format);

  // get impl
  auto input_impl = getMluTensorImpl(input_contiguous);
  auto other_impl = getMluTensorImpl(other_contiguous);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  // create input desc
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOther;
  auto layout = suggest_cnnl_layout(input_tensor);
  descInput.set(input_contiguous, layout);
  descOther.set(other_contiguous, layout);
  // create output desc
  CnnlTensorDescriptor descOutput;
  descOutput.set(output, layout);
  // allocate mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // get workspace size
  size_t tmp_size;
  TORCH_CNNL_CHECK(cnnlGetMinimumWorkspaceSize(handle, descOutput.desc(), &tmp_size));
  auto workspace = at::empty(tmp_size, input.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  // call cnnl min api
  auto workspace_ptr = workspace_impl->cnnlMalloc();
  TORCH_CNNL_CHECK(cnnlMinimum(handle, descInput.desc(), input_ptr, descOther.desc(), other_ptr,
                               descOutput.desc(), output_ptr, workspace_ptr, tmp_size));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
