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

inline std::vector<int64_t> get_other_shape(const at::Tensor& self) {
  std::vector<int64_t> shape = {1};
  if (self.dim() == 0) shape = self.sizes().vec();
  return shape;
}

at::Tensor cnnl_eq(const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_EQ);
}

at::Tensor cnnl_eq(const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_eq(self, other_tensor);
}

at::Tensor& cnnl_eq_(at::Tensor& self, const at::Tensor& other) {
  TORCH_MLU_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  return cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_EQ);
}

at::Tensor& cnnl_eq_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_eq_(self, other_tensor);
}

at::Tensor& cnnl_eq_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_EQ);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_eq_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_eq_out(out , self, other_tensor);
}

at::Tensor cnnl_le(const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_LE);
}

at::Tensor cnnl_le(const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_le(self, other_tensor);
}

at::Tensor& cnnl_le_(at::Tensor& self, const at::Tensor& other) {
  cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_LE);
  return self;
}

at::Tensor& cnnl_le_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_le_(self, other_tensor);
}

at::Tensor& cnnl_le_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_LE);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_le_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_le_out(out, self, other_tensor);
}

at::Tensor cnnl_ge(const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape,
          self.options().dtype(at::kBool),
          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_GE);
}

at::Tensor cnnl_ge(const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_ge(self, other_tensor);
}

at::Tensor& cnnl_ge_(at::Tensor& self, const at::Tensor& other) {
  cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_GE);
  return self;
}

at::Tensor& cnnl_ge_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_ge_(self, other_tensor);
}

at::Tensor& cnnl_ge_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_GE);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_ge_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_ge_out(out, self, other_tensor);
}

at::Tensor cnnl_ne(const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_NE);
}

at::Tensor cnnl_ne(const at::Tensor& self, at::Scalar other) {
  at::Tensor other_tensor = at::full(get_other_shape(self),
        other, self.options().dtype(self.scalar_type()).device(at::kMLU));
  return cnnl_ne(self, other_tensor);
}

at::Tensor& cnnl_ne_(at::Tensor& self, const at::Tensor& other) {
  cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_NE);
  return self;
}

at::Tensor& cnnl_ne_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_ne_(self, other_tensor);
}

at::Tensor& cnnl_ne_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_NE);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor& cnnl_ne_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_ne_out(out, self, other_tensor);
}

at::Tensor cnnl_gt(const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_GT);
}

at::Tensor cnnl_gt(const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_gt(self, other_tensor);
}

at::Tensor& cnnl_gt_(at::Tensor& self, const at::Tensor& other) {
  cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_GT);
  return self;
}

at::Tensor& cnnl_gt_(at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_gt_(self, other_tensor);
}

at::Tensor & cnnl_gt_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_GT);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor & cnnl_gt_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_gt_out(out, self, other_tensor);
}

at::Tensor cnnl_lt(const at::Tensor &self, const at::Tensor &other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  return cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_LT);
}

at::Tensor cnnl_lt(const at::Tensor &self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_lt(self, other_tensor);
}

at::Tensor & cnnl_lt_(at::Tensor &self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_logic_internal(self, self, other_tensor, CNNL_LOGIC_OP_LT);
}

at::Tensor & cnnl_lt_(at::Tensor &self, const at::Tensor &other) {
  return cnnl_logic_internal(self, self, other, CNNL_LOGIC_OP_LT);
}

at::Tensor & cnnl_lt_out(at::Tensor &out, const at::Tensor &self, const at::Tensor &other) {
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output = at::empty(output_shape, self.options().dtype(at::kBool),
                          self.suggest_memory_format());
  cnnl_logic_internal(output, self, other, CNNL_LOGIC_OP_LT);
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor & cnnl_lt_out(at::Tensor &out, const at::Tensor &self, at::Scalar other) {
  auto other_tensor = at::full(get_other_shape(self), other, self.options());
  return cnnl_lt_out(out, self, other_tensor);
}

bool cnnl_equal(const at::Tensor& self, const at::Tensor& other) {
  if (self.sizes() != other.sizes()) {
    return false;
  }
  at::Tensor output;
  auto output_shape = at::infer_size(self.sizes(), other.sizes());
  auto output_tensor = at::empty(output_shape, self.options().dtype(at::kBool),
                                self.suggest_memory_format());
  cnnl_logic_internal(output_tensor, self, other, CNNL_LOGIC_OP_EQ);
  output = cnnl_all(output_tensor);
  auto output_scalar = output.item();
  return output_scalar.to<bool>();
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
