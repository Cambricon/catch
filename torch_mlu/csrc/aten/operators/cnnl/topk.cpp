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

// cnnl_topk_out interface handle memory_format
std::tuple<at::Tensor, at::Tensor> cnnl_topk(const at::Tensor& self,
                                             int64_t k, int64_t dim,
                                             bool largest, bool sorted) {
  at::Tensor values = at::empty({0}, self.options());
  at::Tensor indices = at::empty({0}, self.options().dtype(at::kLong));
  cnnl_topk_out(values, indices, self, k, dim, largest, sorted);
  return std::make_tuple(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_topk_out(at::Tensor& values,
                                                   at::Tensor& indices,
                                                   const at::Tensor& self,
                                                   int64_t k,
                                                   int64_t dim,
                                                   bool largest,
                                                   bool sorted) {
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  TORCH_MLU_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  TORCH_MLU_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isIntegralType(self.scalar_type(), /*includeBool=*/false),
      "cnnl_topk is not implemented for ",
      self.scalar_type());
  at::native::_allocate_or_resize_output_with_indices(values, indices, self, dim, k);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  if (self.numel() == 0) {
    return std::forward_as_tuple(values, indices);
  }

  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto values_contiguous = cnnl_contiguous(values, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  // TODO(daitian): convert integral type to float type because of cnnl limits
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    auto self_pre_cast = at::empty(self_contiguous.sizes(),
                                   self_contiguous.options().dtype(at::ScalarType::Float),
                                   memory_format);
    auto values_pre_cast = at::empty(values_contiguous.sizes(),
                                     values_contiguous.options().dtype(at::ScalarType::Float),
                                     memory_format);
    cnnl_cast(self_contiguous, self_pre_cast);
    cnnl_cast(values_contiguous, values_pre_cast);
    cnnl_topk_internal(values_pre_cast, indices_contiguous,
                       self_pre_cast, k, dim, largest, sorted);
    auto values_post_cast = at::empty(
        values_contiguous.sizes(), values_contiguous.options(), memory_format);
    cnnl_cast(values_pre_cast, values_post_cast);
    values_contiguous = values_post_cast;
  } else {
    cnnl_topk_internal(values_contiguous, indices_contiguous,
                       self_contiguous, k, dim, largest, sorted);
  }

  getMluTensorImpl(values)->copy_cnnl_metadata_from(getMluTensorImpl(values_contiguous));
  getMluTensorImpl(indices)->copy_cnnl_metadata_from(getMluTensorImpl(indices_contiguous));
  resize_impl_mlu_(getMluTensorImpl(values), values_contiguous.sizes(),
                   values_contiguous.strides());
  resize_impl_mlu_(getMluTensorImpl(indices), indices_contiguous.sizes(),
                   indices_contiguous.strides());
  return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> cnnl_sort(const at::Tensor& input, int64_t dim,
                                             bool descending = false) {
  dim = at::maybe_wrap_dim(dim, input.dim(), /*wrap_scalar=*/true);
  if (input.dim() == 0) {
    return std::make_tuple(input, at::zeros({}, input.options().dtype(at::kLong)));
  }
  auto memory_format = input.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  // TODO(guyi): Stuff
  // here use two cast to deal with the case where input dtype is not fp32
  // later will remove them when cnnlTopk support int32
  if (at::isIntegralType(input.scalar_type(), /*includeBool=*/false)) {
    at::Tensor input_cast = at::empty(
      input_contiguous.sizes(),
      input_contiguous.options().dtype(at::ScalarType::Float),
      memory_format);
    cnnl_cast(input_contiguous, input_cast);
    auto output = cnnl_sort_internal(input_cast, dim, descending, true);
    at::Tensor output_cast = at::empty(
      input_contiguous.sizes(), input_contiguous.options(), memory_format);
    cnnl_cast(std::get<0>(output), output_cast);
    return std::make_tuple(output_cast, std::get<1>(output));
  } else {
    return cnnl_sort_internal(input_contiguous, dim, descending, true);
  }
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_sort_out(at::Tensor& values, at::Tensor& indices,
                                                   const at::Tensor& input, int64_t dim,
                                                   bool descending = false) {
  auto output = cnnl_sort(input, dim, descending);
  auto values_mlu = std::get<0>(output);
  auto indices_mlu = std::get<1>(output);
  getMluTensorImpl(values)->copy_cnnl_metadata_from(getMluTensorImpl(values_mlu));
  getMluTensorImpl(indices)->copy_cnnl_metadata_from(getMluTensorImpl(indices_mlu));
  resize_impl_mlu_(getMluTensorImpl(values), values_mlu.sizes(), values_mlu.strides());
  resize_impl_mlu_(getMluTensorImpl(indices), indices_mlu.sizes(), indices_mlu.strides());
  return std::forward_as_tuple(values, indices);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
