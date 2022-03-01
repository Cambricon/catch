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
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_div_out_internal(at::Tensor& output,
                                  const at::Tensor& input,
                                  const at::Tensor& other) {
  auto output_type = output.scalar_type();
  auto compute_type = get_compute_type(input, other, output);
  auto output_contiguous = cnnl_contiguous(output, output.suggest_memory_format());
  auto input_new = convertTensorType(input, compute_type);
  auto other_new = convertTensorType(other, compute_type);
  auto output_new = convertTensorType(output_contiguous, compute_type);

  // get tensor size and stride based on memory format
  auto memory_format = output.suggest_memory_format();
  auto output_size_stride = get_tensor_size_stride(output_new, memory_format);
  auto input_size_stride = get_tensor_size_stride(input_new, memory_format);
  auto other_size_stride = get_tensor_size_stride(other_new, memory_format);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_other;
  CnnlTensorDescriptor desc_output;

  // get cnnl descriptor
  desc_input.set(input_new, std::get<0>(input_size_stride),
                 std::get<1>(input_size_stride), CNNL_LAYOUT_ARRAY);
  desc_other.set(other_new, std::get<0>(other_size_stride),
                 std::get<1>(other_size_stride), CNNL_LAYOUT_ARRAY);
  desc_output.set(output_new, std::get<0>(output_size_stride),
                  std::get<1>(output_size_stride), CNNL_LAYOUT_ARRAY);


  auto input_impl = getMluTensorImpl(input_new);
  auto other_impl = getMluTensorImpl(other_new);
  auto output_impl = getMluTensorImpl(output_new);

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  if (input_impl->numel() == 0)
      return output;

  // workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetDivWorkspaceSize(handle, desc_input.desc(), desc_other.desc(),
                              desc_output.desc(), &workspace_size));
  size_t workspace_itemnum = workspace_size / input_impl->itemsize();
  at::Tensor temp =
      at::empty({static_cast<long int>(workspace_itemnum)}, input.options());
  auto* temp_impl = getMluTensorImpl(temp);
  auto temp_ptr = temp_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlDiv(handle, desc_input.desc(), input_ptr,
                           desc_other.desc(), other_ptr, temp_ptr, workspace_size,
                           desc_output.desc(), output_ptr));
  if (compute_type != output_type) {
    cnnl_cast_internal(output_new, output);
  }
  if (!output.is_contiguous(c10::MemoryFormat::Contiguous)
          || output_type != compute_type) {
    output.copy_(output_contiguous);
  }
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
