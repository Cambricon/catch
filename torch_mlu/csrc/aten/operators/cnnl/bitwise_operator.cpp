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
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& bitwise_operator_out(at::Tensor& out, const at::Tensor& self,
                                      const at::Tensor& other, cnnlBitComputeOp_t op) {
  // XXX(liuyuxin): the algo to determine common type is different
  // from the algo using in TensorIterator::compute_types,
  // which may cause unexpected error when comparsion between MLU op and CPU op
  // Also, some type conversions on MLU are not supported, please check
  // torch_mlu/cstc/aten/operators/cnnl/internal/cast_internal.cpp
  auto common_type = at::result_type(self, other);
  at::Tensor self_tmp = self;
  at::Tensor other_tmp = other;
  if (self.scalar_type() != common_type) {
    // FIXME(liuyuxin): inplace op while cause storage assert failed in
    // DEBUG mode : pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:1585
    self_tmp = self.to(common_type);
    out = out.to(common_type);
  }
  if (other.scalar_type() != common_type) {
    other_tmp = other.to(common_type);
  }

  // broadcast uneven dims
  auto shape_broadcast = broadcast_shape(self_tmp, other_tmp);
  if (self_tmp.dim() > other_tmp.dim()) {
    other_tmp = other_tmp.reshape(std::get<1>(shape_broadcast));
  } else if (self_tmp.dim() < other_tmp.dim()) {
    self_tmp = self_tmp.reshape(std::get<0>(shape_broadcast));
  }
  auto infer_out = at::infer_size(self_tmp.sizes(), other_tmp.sizes());
  auto numel_ = self_tmp.numel();
  auto out_impl = getMluTensorImpl(out);
  auto memory_format = self_tmp.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self_tmp, memory_format);
  auto other_contiguous = cnnl_contiguous(other_tmp, memory_format);
  if (out.numel() >= numel_) {
    if (self_contiguous.numel() >= other_contiguous.numel()) {
      resize_impl_mlu_(out_impl, self_contiguous.sizes(), self_contiguous.strides());
      cnnl_bitwise_op_out_internal(out, self_contiguous, other_contiguous, op);
    } else {
      resize_impl_mlu_(out_impl, other_contiguous.sizes(), other_contiguous.strides());
      cnnl_bitwise_op_out_internal(out, other_contiguous, self_contiguous, op);
    }
    return out;
  }
  auto output = at::empty(infer_out, self_contiguous.options());
  auto stride = self_contiguous.strides();
  if (self_contiguous.numel() >= other_contiguous.numel()) {
    cnnl_bitwise_op_out_internal(output, self_contiguous, other_contiguous, op);
  } else {
    stride = other_contiguous.strides();
    cnnl_bitwise_op_out_internal(output, other_contiguous, self_contiguous, op);
  }
  out_impl->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(out_impl, output.sizes(), stride);
  return out;
}

at::Tensor& bitwise_operator_out(at::Tensor& out, const at::Tensor &self,
                                      at::Scalar other, cnnlBitComputeOp_t op) {
  auto other_data = other.to<float>();
  auto type_ = at::kLong;
  if (other.isIntegral()) type_ = at::kInt;
  else if (other.isBoolean()) type_ = at::kBool;
  auto other_tensor =
      at::full({}, other_data, self.options().dtype(type_).device(at::kMLU));
  bitwise_operator_out(out, self, other_tensor, op);
  return out;
}


at::Tensor& cnnl_bitwise_or_out(at::Tensor& out,
                                const at::Tensor& self,
                                const at::Tensor& other) {
  return bitwise_operator_out(out, self, other, CNNL_CYCLE_BOR_OP);
}

at::Tensor& cnnl_bitwise_or_out(at::Tensor& out,
                                const at::Tensor& self,
                                at::Scalar other) {
  return bitwise_operator_out(out, self, other, CNNL_CYCLE_BOR_OP);
}

at::Tensor& cnnl_bitwise_and_out(at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& other) {
  return bitwise_operator_out(out, self, other, CNNL_CYCLE_BAND_OP);
}

at::Tensor& cnnl_bitwise_and_out(at::Tensor& out, const at::Tensor &self,
                                at::Scalar other) {
  return bitwise_operator_out(out, self, other, CNNL_CYCLE_BAND_OP);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
