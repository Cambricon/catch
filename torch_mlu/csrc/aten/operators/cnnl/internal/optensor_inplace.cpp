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

#include "aten/util/dispatch.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_optensor_inplace_internal(at::Tensor& input,
                                          const at::Tensor& other,
                                          at::Scalar beta_scalar,
                                          at::Scalar alpha_scalar,
                                          cnnlOpTensorDesc_t op_type) {
  auto memory_format = input.suggest_memory_format();
  std::vector<int64_t> other_size = other.sizes().vec();
  const int other_dim = other.dim();
  const int input_dim = input.dim();
  at::Tensor other_contiguous = cnnl_contiguous(other, memory_format);
  // broadcast tensors size, and set tensor info by new size and stride.
  auto shape_broadcast = broadcast_shape(input, other_contiguous);

  // get output shape from input shapes and check input shape with output shape.
  std::vector<int64_t> output_shape = at::infer_size(input.sizes(),
                                                     other_contiguous.sizes());
  TORCH_MLU_CHECK((output_shape == input.sizes())
                  && (std::get<0>(shape_broadcast) == input.sizes()),
    "For inplace op, output tensor shape need be equal input tensor shape.");

  auto other_tensor = convertScalarToTensor(other_contiguous, input.scalar_type());
  other_tensor = other_tensor.to(input.dtype());
  if (input.is_contiguous(memory_format)) {
    return cnnl_optensor_compute(input,
                                 input,
                                 other_tensor,
                                 alpha_scalar,
                                 beta_scalar,
                                 op_type,
                                 input.scalar_type());
  }
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor other_desc;

  // modify Tensor shape and stride based on broadcast shape.
  auto set_desc_by_broadcast_shape_info = [](at::Tensor& self,
                          CnnlTensorDescriptor& self_desc,
                          const std::vector<int64_t>& shape,
                          const cnnlDataType_t dtype) -> void {
    auto stride = self.strides().vec();
    const int diff = shape.size() - self.dim();
    TORCH_MLU_CHECK(diff >= 0, "Infer shape size need greater than tensor dim.");
    int64_t value = 1;
    if (stride.size() > 0) {
      value = stride.front();
    }
    stride.insert(stride.begin(), diff, value);
    self_desc.set(self, shape, stride,
                  CNNL_LAYOUT_ARRAY, dtype);
  };
  auto dtype = getCnnlDataType(input.dtype());
  set_desc_by_broadcast_shape_info(input, input_desc,
                                   std::get<0>(shape_broadcast),
                                   dtype);
  set_desc_by_broadcast_shape_info(other_tensor, other_desc,
                                   std::get<1>(shape_broadcast),
                                   dtype);

  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto other_impl = getMluTensorImpl(other_tensor);
  auto input_ptr = input_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();

  // workspace
  size_t workspace_size = 0;
  void* temp_ptr = nullptr;
  at::Tensor temp;
  TORCH_CNNL_CHECK(
      cnnlGetOpTensorWorkspaceSize(handle, input_desc.desc(), other_desc.desc(),
                                   input_desc.desc(), &workspace_size));
  if (workspace_size != 0) {
    temp = at::empty({static_cast<long int>(workspace_size)},
                                input.options().dtype(at::kByte));
    auto* temp_impl = getMluTensorImpl(temp);
    temp_ptr = temp_impl->cnnlMalloc();
  }

  CnnlOpTensorDescriptor descOpTensor;
  descOpTensor.set(op_type, input_impl->getCnnlType(), CNNL_NOT_PROPAGATE_NAN);

  float out_k = 0.0;
  auto alpha = alpha_scalar.to<float>();
  auto beta = beta_scalar.to<float>();

  /* C = op(alpha1 * C, alpha2 * B) + out_k * C */
  AT_DISPATCH_MLU_OPTENSOR_TYPES_AND_HALF(
      input.scalar_type(), "optensor_inplace", [&] {
        auto alpha_value = static_cast<scalar_t>(alpha);
        auto beta_value = static_cast<scalar_t>(beta);
        auto out_value = static_cast<scalar_t>(out_k);
        TORCH_CNNL_CHECK(cnnlOpTensor(
            handle, descOpTensor.desc(), &alpha_value, input_desc.desc(),
            input_ptr, &beta_value, other_desc.desc(), other_ptr, temp_ptr,
            workspace_size, &out_value, input_desc.desc(), input_ptr));
      });
  return input;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
