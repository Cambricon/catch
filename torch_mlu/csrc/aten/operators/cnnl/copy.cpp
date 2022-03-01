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

#include "aten/core/tensor_impl.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_copy_(at::Tensor& self, const at::Tensor& src,
                       bool non_blocking) {
  TORCH_MLU_CHECK(self.sizes() == src.sizes(),
    "self sizes should equal to src sizes in copy_.");
  if (self.numel() == 0) {
    return self;
  }
  // D2D: src (MLU) => self (MLU)
  if (is_mlu(self) && is_mlu(src)) {
    torch_mlu::mlu::MLUGuard guard(src.device());
    if ((self.dtype() == src.dtype())
        && (self.device().index() == src.device().index())) {
      if (self.is_non_overlapping_and_dense()) {
        non_overlapping_and_dense_out(self, src);
      } else {
        cnnl_copy_internal(self, src);
      }
    } else {
      cnnl_cast_internal(src, self);
    }
    return self;
  }
  // H2D: src (CPU) => self (MLU)
  // 1. Copy the data form the CPU to the MLU device.
  // 2. If data transformation is needed. it will be done by
  //    an MLU Transpose operator.
  // 3. Copy the data onto the self tensor.
  if (is_mlu(self)) {
    // cnrtMemcpy don't suppoert stride.
    auto memory_format = self.suggest_memory_format();
    at::Tensor src_non_overlapping_and_dense;
    if (src.is_non_overlapping_and_dense()) {
      src_non_overlapping_and_dense = src;
    } else {
      src_non_overlapping_and_dense = src.contiguous(memory_format);
    }
    // cnnlMalloc call memCopyAsyn when self data_ptr is nullptr.
    if ((self.strides() == src_non_overlapping_and_dense.strides())
        && self.data_ptr() == nullptr
        && self.dtype() == src.dtype()) {
      copy_from_cpu(self, src_non_overlapping_and_dense, non_blocking,
                    memory_format);
    } else {
      auto src_mlu = at::empty_like(src_non_overlapping_and_dense,
                                    src_non_overlapping_and_dense.options().device(at::kMLU));
      copy_from_cpu(src_mlu,
                    src_non_overlapping_and_dense,
                    non_blocking,
                    memory_format);
      if (self.dtype() == src.dtype()) {
        cnnl_copy_internal(self, src_mlu);
      } else {
        cnnl_cast_internal(src_mlu, self);
      }
    }

    auto queue = getCurrentQueue();
    if (!non_blocking) queue.synchronize();
    return self;
  }

  // D2H: src (MLU) => self (CPU)
  if (non_blocking) {
    CNLOG(WARNING) << "Copy MLU data back to CPU device requires "
          << "synchronization to ensure correct data, "
          << "non_blocking parameter is invalid.";
  }
  // cnrtMemcpy don't suppoert stride.
  auto memory_format = self.suggest_memory_format();
  at::Tensor src_non_overlapping_and_dense;
  if (src.is_non_overlapping_and_dense()) {
    src_non_overlapping_and_dense = src;
  } else {
    src_non_overlapping_and_dense = src.contiguous(memory_format);
  }
  // Note: optimize performance for D2H copy
  // Comparing to at::empty,
  // at::zeros initialize real memory immediately and get better performance.
  auto src_cpu = at::empty_like(src_non_overlapping_and_dense,
                                src_non_overlapping_and_dense.options().device(at::kCPU)).zero_();
  copy_to_cpu_cnnl(src_cpu, src_non_overlapping_and_dense, memory_format);
  self.copy_(src_cpu);
  return self;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
