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

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "ATen/native/BinaryOps.h"
#include "ATen/native/TensorIterator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

#include "aten/core/DispatchStub.h"

using at::Tensor;
using at::native::DispatchStub;
using at::native::add_stub;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_add(const at::Tensor& input, const at::Tensor& other,
                    at::Scalar alpha_scalar) {
  return at::native::add(input, other, alpha_scalar);
}

at::Tensor cnnl_add(const at::Tensor& input, at::Scalar other,
                    at::Scalar alpha_scalar) {
  return at::native::add(input, other, alpha_scalar);
}

at::Tensor& cnnl_add_(at::Tensor& self, at::Scalar other, at::Scalar alpha) {
  return at::native::add_(self, other, alpha);
}

at::Tensor& cnnl_add_(at::Tensor& self, const at::Tensor& other,
                      at::Scalar alpha) {
  return at::native::add_(self, other, alpha);
}

at::Tensor& cnnl_add_out(at::Tensor& out, const at::Tensor& self,
    const at::Tensor& other, at::Scalar alpha) {
  return at::native::add_out(out, self, other, alpha);
}

void add_mlu_kernel(at::TensorIterator& iter, at::Scalar scalar) {
  auto output = iter.output(0);
  std::vector<at::Tensor> list;
  get_contiguous(iter, list);
  // add has 2 input tensors.
  auto self = list[0];
  auto other = list[1];
  if (isCpuScalar(other) && isTransformComputeType(self)) {
    auto other_data = other.item().to<float>() * scalar.to<float>();
    cnnl_transform_out_internal(output, self, 1, other_data);
    return;
  }
  if (isCpuScalar(self) && isTransformComputeType(other)) {
    cnnl_transform_out_internal(output, other, scalar.to<float>(), self.item());
    return;
  }
  auto self_mlu = convertScalarToTensor(self, self.scalar_type());
  auto other_mlu = convertScalarToTensor(other, other.scalar_type());
  cnnl_optensor_out_internal(output, self_mlu, other_mlu, 1, scalar, CNNL_OP_TENSOR_ADD);
}
REGISTER_MLU_DISPATCH(add_stub, &add_mlu_kernel);



}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
