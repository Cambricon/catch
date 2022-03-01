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

namespace torch_mlu {
namespace cnnl {
namespace ops {

static inline void checkSetStorage(at::Tensor& result, c10::Storage storage, int64_t storage_offset,
                                   at::IntArrayRef size, at::IntArrayRef stride) {
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "unequal size length (", size.size(),
                                              ") and stride length (", stride.size(), ")");
  }

  // storage: note this can't be replaced with result.set_(storage) as the semantics of that
  // function is to set the tensor size to be equal to the size of the storage.
  if (!result.storage().is_alias_of(storage)) {
    TORCH_CHECK(storage, "Caffe2 might have tensors whose storages are null, "
      "but we don't allow it in PyTorch.");
    TORCH_CHECK(result.storage(), "Caffe2 might have tensors whose storages are null, "
      "but we don't allow it in PyTorch.");

    // We used to allow this, but this breaks device caching.
    // Let's put an actual error message for this one.
    TORCH_CHECK(result.storage().device() == storage.device(),
                "Attempted to set the storage of a tensor on device \"", result.storage().device(),
                "\" to a storage on different device \"", storage.device(),
                "\".  This is no longer allowed; the devices must match.");
    getMluTensorImpl(result)->set_storage_keep_dtype(storage);
  }

  TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
}

// Be careful!! Can not share memory because of MLU layout, currently only used by DDP
// （empty + set_.source_Storage_storage_offset）can be used to create a Tensor sharing
// storage with other Tensor.
at::Tensor& cnnl_set_storage_(at::Tensor &self, c10::Storage source, int64_t storage_offset,
  at::IntArrayRef size, at::IntArrayRef stride) {
  checkSetStorage(self, source, storage_offset, size, stride);

  getMluTensorImpl(self)->set_storage_offset(storage_offset);
  c10::optional<at::IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          c10::optional<at::IntArrayRef>(stride) : c10::nullopt;
  resize_impl_mlu_(getMluTensorImpl(self), size, stride_opt);
  return self;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
