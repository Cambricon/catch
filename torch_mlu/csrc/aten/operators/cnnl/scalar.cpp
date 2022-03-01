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

#include <ATen/Dispatch.h>

#include "aten/device/queue.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Scalar cnnl_local_scalar_dense(const at::Tensor& self) {
  at::Scalar r;
  auto tensor_impl = getMluTensorImpl(self);
  auto tensor_ptr = tensor_impl->cnnlMalloc();
  auto queue = getCurrentQueue();

  // Data representation of type long is only supported using int.
  if (self.scalar_type() == at::ScalarType::Long) {
    int value;
    cnrtMemcpyAsync(&value, tensor_ptr, sizeof(int), queue.queue(), CNRT_MEM_TRANS_DIR_DEV2HOST);
    queue.synchronize();
    return at::Scalar(static_cast<long>(value));
  }

  // local_scalar_dense
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(),
          "_local_scalar_dense", [&] {
      scalar_t value;
      cnrtMemcpyAsync(&value, tensor_ptr, sizeof(scalar_t), queue.queue(),
                      CNRT_MEM_TRANS_DIR_DEV2HOST);
      queue.synchronize();
      r = at::Scalar(value);
  });
  return r;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
