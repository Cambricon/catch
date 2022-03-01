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

#include <c10/util/Optional.h>

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include "ATen/InferSize.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_alias_with_sizes_and_strides(const at::Tensor& self,
                                             const c10::IntArrayRef sizes,
                                             const c10::IntArrayRef strides) {
  at::Tensor self_;
  auto impl = c10::make_intrusive<MLUTensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(sizes, strides);
  self_ = at::Tensor(std::move(impl));
  return self_;
}

at::Tensor cnnl_view(const at::Tensor &self, at::IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  if ((!stride.has_value())
      && (self.dim() < 6) && (self.dim() > 3)
      && (self.is_contiguous(get_channels_last_memory_format(self.dim())))) {
    auto self_channels_first = permute_to_contiguous(self, c10::MemoryFormat::Contiguous);
    inferred_size = at::infer_size(size, self_channels_first.numel());
    stride = at::detail::computeStride(self_channels_first.sizes(),
                                       self_channels_first.strides(),
                                       inferred_size);
    auto stride_value = *stride;
    return cnnl_alias_with_sizes_and_strides(self_channels_first,
                                             inferred_size,
                                             stride_value);
  }
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  auto* input_impl = getMluTensorImpl(self);
  auto output = cnnl_alias_with_sizes_and_strides(self, inferred_size, stride_value);
  auto* output_impl = getMluTensorImpl(output);
  output_impl->insert_views_op_info(VIEWOPNAME::view, input_impl,
                                    self.is_contiguous());
  return output;
}



at::Tensor cnnl_reshape(const at::Tensor &self, at::IntArrayRef shape) {
  auto* input_impl = getMluTensorImpl(self);
  auto output = at::native::reshape(self, shape);
  auto* output_impl = getMluTensorImpl(output);
  output_impl->insert_views_op_info(VIEWOPNAME::reshape, input_impl,
                                    self.is_contiguous());
  return output;
}

at::Tensor& cnnl_resize_(at::Tensor &self, at::IntArrayRef size,
                         c10::optional<c10::MemoryFormat> memory_format) {
  resize_impl_mlu_(getMluTensorImpl(self), size, /*strides=*/c10::nullopt);
  return self;
}

at::Tensor cnnl_view(const at::Tensor & input, c10::ScalarType dtype) {
    if (input.scalar_type() == dtype) {
        return input;
    }
    auto type_meta = c10::scalarTypeToTypeMeta(dtype);
    TORCH_CHECK(input.element_size() == static_cast<int64_t>(type_meta.itemsize()),
            "Viewing a tensor as a new dtype with a different ",
            "number of bytes per element is not supported.");
    c10::Storage storage = input.storage();
    auto new_tensor = at::detail::make_tensor<MLUTensorImpl>(
        std::move(storage), input.key_set(), type_meta);
    auto* impl = getMluTensorImpl(new_tensor);
    impl->set_storage_offset(input.storage_offset());
    impl->set_sizes_and_strides(input.sizes(), input.strides());
    return new_tensor;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
