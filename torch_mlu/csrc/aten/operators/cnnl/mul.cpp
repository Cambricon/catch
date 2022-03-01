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
using at::native::mul_stub;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_mul(const at::Tensor& self, const at::Tensor& other) {
  return at::native::mul(self, other);
}

at::Tensor cnnl_mul(const at::Tensor& self, at::Scalar other) {
  return at::native::mul(self, at::scalar_to_tensor(other));
}

at::Tensor& cnnl_mul_(at::Tensor& self, const at::Tensor& other) {
  return  at::native::mul_(self, other);
}

at::Tensor& cnnl_mul_(at::Tensor& self, at::Scalar other) {
  return  at::native::mul_(self, other);
}

at::Tensor& cnnl_mul_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  return at::native::mul_out(out, self, other);
}

void mul_mlu_kernel(at::TensorIterator& iter) {
  auto output = iter.output(0);
  std::vector<at::Tensor> list;
  get_contiguous(iter, list);
  // mul has 2 input tensors.
  auto self = list[0];
  auto other = list[1];
  if (isCpuScalar(other) && isTransformComputeType(self)) {
    cnnl_transform_out_internal(output, self, other.item(), 0);
    return;
  }
  if (isCpuScalar(self) && isTransformComputeType(other)) {
    cnnl_transform_out_internal(output, other, self.item(), 0);
    return;
  }
  auto self_mlu = convertScalarToTensor(self, self.scalar_type());
  auto other_mlu = convertScalarToTensor(other, other.scalar_type());
  cnnl_optensor_out_internal(output, self_mlu, other_mlu, 1, 1, CNNL_OP_TENSOR_MUL);
}
REGISTER_MLU_DISPATCH(mul_stub, &mul_mlu_kernel);


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
