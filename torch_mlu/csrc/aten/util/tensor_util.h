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

#pragma once
#include <torch/csrc/jit/ir/ir.h>
#include <ATen/TensorUtils.h>
#include "aten/core/tensor_impl.h"
#include "aten/device/device.h"

namespace torch_mlu {

MLUTensorImpl* resize_impl_mlu_(MLUTensorImpl* self, at::IntArrayRef size,
                                c10::optional<at::IntArrayRef> stride);

MLUTensorImpl* getMluTensorImpl(const at::Tensor& tensor);

void copy_to_cpu_cnnl(at::Tensor& dst, const at::Tensor& src,
                      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

void copy_from_cpu(at::Tensor& dst, const at::Tensor& src, bool non_blocking = false,
                   c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

void ShareCpuData(at::Tensor& dst, const at::Tensor& src);

bool is_mlu(const at::Tensor& t);

bool is_mlu(const c10::TensorImpl* t);

bool is_channels_last(const at::Tensor& t);

std::vector<int64_t> get_channels_last_strides_1d(const at::IntArrayRef sizes);

std::vector<int64_t> get_channels_last_strides(const at::IntArrayRef sizes);

std::vector<int64_t> get_channels_first_strides(const at::IntArrayRef sizes);

std::vector<int64_t> get_contiguous_strides(const at::IntArrayRef sizes,
             c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

void checkAllSameMLU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors);

void checkSameMLU(at::CheckedFrom c, const at::TensorArg& t1, const at::TensorArg& t2);

inline int getTensorDevice(at::TensorList list) {
    int device_index = current_device();
    for (auto& t : list) {
        if (t.device().type() == at::kMLU) {
            device_index = t.device().index();
            break;
        }
    }
    return device_index;
}

void resize_mlu_storage_bytes(at::StorageImpl* self, ptrdiff_t size_bytes,
                              const caffe2::TypeMeta& dtype_offchip);

static inline void maybe_resize_storage_mlu(MLUTensorImpl* self, int64_t new_size) {
  if (new_size > 0) {
    TORCH_MLU_CHECK(self->storage(), "Cannot use PyTorch operations on a half-constructed "
      "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
      "it first; otherwise, this is a bug, please report it.");
    TORCH_MLU_CHECK(self->storage().unsafeGetStorageImpl(),
      "maybe_resize_storage_mlu failed, invalid null storage");
    uint64_t new_size_bytes = (new_size + self->storage_offset()) * self->dtype().itemsize();
    if (new_size_bytes > self->storage().nbytes()) {
      resize_mlu_storage_bytes(self->storage().unsafeGetStorageImpl(), new_size_bytes,
        self->dtype());
    }
  }
}

}  // namespace torch_mlu
