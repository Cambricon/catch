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


at::Tensor& cnnl_masked_select_internal(at::Tensor& output,
                                        const at::Tensor& input,
                                        const at::Tensor& mask) {
  auto num_output = at::empty({1}, output.options().dtype(at::kByte));
  uint32_t num =  0;
  auto output_impl = getMluTensorImpl(output);
  auto input_impl = getMluTensorImpl(input);
  auto mask_impl = getMluTensorImpl(mask);
  auto num_output_impl = getMluTensorImpl(num_output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  CnnlTensorDescriptor desc_mask;
  desc_input.set(input, CNNL_LAYOUT_ARRAY);
  desc_mask.set(mask, CNNL_LAYOUT_ARRAY);
  desc_output.set(output);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto mask_ptr = mask_impl->cnnlMalloc();
  auto num_output_ptr = num_output_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlUnarySelect(handle,
                                   desc_input.desc(),
                                   input_ptr,
                                   desc_mask.desc(),
                                   mask_ptr,
                                   desc_output.desc(),
                                   output_ptr,
                                   static_cast<uint32_t *>(num_output_ptr)));

   // add synchronization point to recive num_output
   auto queue = getCurrentQueue();
   TORCH_CNRT_CHECK(cnrtMemcpyAsync(&num,
                               num_output_ptr,
                               sizeof(uint32_t),
                               queue.queue(),
                               CNRT_MEM_TRANS_DIR_DEV2HOST));
   queue.synchronize();
   resize_impl_mlu_(getMluTensorImpl(output), {num}, c10::nullopt);
   return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
