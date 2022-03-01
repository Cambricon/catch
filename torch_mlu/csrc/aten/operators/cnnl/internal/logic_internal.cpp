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
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_logic_internal(at::Tensor& output,
                               const at::Tensor& input,
                               const at::Tensor& other,
                               cnnlLogicOp_t logic_type) {
  TORCH_CHECK(input.dim() <= 8 && other.dim() <= 8, "dimension not support");

  at::Tensor input_tmp = input;
  at::Tensor other_tmp = other;
  // broadcast input and other
  auto shape_broadcast = broadcast_shape(input, other);
  if (input.dim() > other.dim()) {
    other_tmp = other_tmp.reshape(std::get<1>(shape_broadcast));
  } else if (input.dim() < other.dim()) {
    input_tmp = input_tmp.reshape(std::get<0>(shape_broadcast));
  }

  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input_tmp, memory_format);
  auto other_contiguous = cnnl_contiguous(other_tmp, memory_format);

  auto input_impl = getMluTensorImpl(input_contiguous);
  auto other_impl = getMluTensorImpl(other_contiguous);
  auto output_impl = getMluTensorImpl(output);

  // get suggest layout
  auto layout = suggest_cnnl_layout(input_contiguous);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor output_desc;

  // Modify size info for create desc
  auto input_size = modify_dims_based_on_layout(input_contiguous.sizes().vec(), memory_format);
  auto other_size = modify_dims_based_on_layout(other_contiguous.sizes().vec(), memory_format);

  input_desc.set(input_contiguous, input_size, get_contiguous_strides(input_size), layout);
  other_desc.set(other_contiguous, other_size, get_contiguous_strides(other_size), layout);
  output_desc.set(output, layout);

  // compute size of workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetLogicOpWorkspaceSize(handle, input_desc.desc(), other_desc.desc(),
                                  output_desc.desc(), &workspace_size));

  // malloc workspace
  auto workspace = at::empty(workspace_size, input.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  void* workspace_ptr;
  if (workspace_size) {
    workspace_ptr = workspace_impl->cnnlMalloc();
  } else {
    workspace_ptr = nullptr;
  }

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLogicOp(handle, logic_type, input_desc.desc(), input_ptr,
                               other_desc.desc(), other_ptr, workspace_ptr,
                               workspace_size, output_desc.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

