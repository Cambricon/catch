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

#include "ATen/native/BinaryOps.h"
#include "ATen/native/TensorIterator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

#include "aten/core/DispatchStub.h"

using at::Tensor;
using at::native::DispatchStub;
using at::native::div_stub;

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_div(const at::Tensor& self, const at::Tensor& other) {
  auto output = at::native::empty_like(self, self.options().device(at::Device::Type::MLU));
  cnnl_div_out(output, self, other);
  return output;
}

at::Tensor& cnnl_div_(at::Tensor& self, const at::Tensor& other) {
  cnnl_div_out(self, self, other);
  return self;
}

at::Tensor & cnnl_div_out(at::Tensor & out, const at::Tensor & self,
        const at::Tensor & other) {
  return at::native::div_out(out, self, other);
}

// For C++ only
at::Tensor cnnl_div(const at::Tensor& self, at::Scalar other) {
  auto out = at::native::empty_like(self, self.options().device(at::Device::Type::MLU));
  cnnl_div_out(out, self, at::scalar_to_tensor(other));
  return out;
}

// For C++ only
at::Tensor& cnnl_div_(at::Tensor& self, at::Scalar other) {
  return cnnl_div_out(self, self, at::scalar_to_tensor(other));
}

void div_mlu_kernel(at::TensorIterator& iter) {
  auto output = iter.output(0);
  std::vector<at::Tensor> list;
  get_contiguous(iter, list);
  auto self = list[0];
  auto other = list[1];
  if (isCpuScalar(other) && isTransformComputeType(self)) {
    cnnl_transform_out_internal(output, self, 1.0 / (other.item().to<float>()), 0);
    return;
  }
  auto self_mlu = convertScalarToTensor(self, self.scalar_type());
  auto other_mlu = convertScalarToTensor(other, other.scalar_type());

  // div has 2 input tensors.
  cnnl_div_out_internal(output, self_mlu, other_mlu);
}

at::Tensor cnnl_true_divide(const at::Tensor & self, const at::Tensor & other) {
  return at::native::true_divide(self, other);
}

at::Tensor & cnnl_true_divide_out(at::Tensor & out,
        const at::Tensor & self, const at::Tensor & other) {
  return at::native::true_divide_out(out, self, other);
}

at::Tensor & cnnl_true_divide_inplace(at::Tensor & self, const at::Tensor & other) {
  return at::native::true_divide_(self, other);
}

REGISTER_MLU_DISPATCH(div_stub, &div_mlu_kernel);

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
