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
#include "aten/util/dispatch.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_optensor_compute(at::Tensor& output,
                                 const at::Tensor& self,
                                 const at::Tensor& other,
                                 at::Scalar alpha_scalar,
                                 at::Scalar beta_scalar,
                                 cnnlOpTensorDesc_t op_type,
                                 at::ScalarType compute_type) {
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor output_desc;
  auto self_contiguous = convertTensorType(self, compute_type);
  auto other_contiguous = convertTensorType(other, compute_type);
  auto output_contiguous = convertTensorType(output, compute_type);

  // get tensor size and stride based on memory format
  auto memory_format = output.suggest_memory_format();
  auto output_size_stride = get_tensor_size_stride(output_contiguous, memory_format);
  auto self_size_stride = get_tensor_size_stride(self_contiguous, memory_format);
  auto other_size_stride = get_tensor_size_stride(other_contiguous, memory_format);
  // get cnnl descriptor
  self_desc.set(self_contiguous, std::get<0>(self_size_stride),
                 std::get<1>(self_size_stride), CNNL_LAYOUT_ARRAY);
  other_desc.set(other_contiguous, std::get<0>(other_size_stride),
                 std::get<1>(other_size_stride), CNNL_LAYOUT_ARRAY);
  output_desc.set(output_contiguous, std::get<0>(output_size_stride),
                  std::get<1>(output_size_stride), CNNL_LAYOUT_ARRAY);

  auto self_impl = getMluTensorImpl(self_contiguous);
  auto other_impl = getMluTensorImpl(other_contiguous);
  auto output_impl = getMluTensorImpl(output_contiguous);


  // malloc mlu memory
  auto self_ptr = self_impl->cnnlMalloc();
  auto other_ptr = other_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // workspace
  size_t workspace_size = 0;
  void * temp_ptr = nullptr;
  at::Tensor temp;
  TORCH_CNNL_CHECK(
      cnnlGetOpTensorWorkspaceSize(handle, self_desc.desc(), other_desc.desc(),
                                   output_desc.desc(), &workspace_size));
  if (workspace_size != 0) {
    temp = at::empty({static_cast<long int>(workspace_size)},
                                self_contiguous.options().dtype(at::kByte));
    auto* temp_impl = getMluTensorImpl(temp);
    temp_ptr = temp_impl->cnnlMalloc();
  }
  CnnlOpTensorDescriptor descOpTensor;
  descOpTensor.set(op_type, getCnnlDataType(c10::scalarTypeToTypeMeta(compute_type)),
                   CNNL_NOT_PROPAGATE_NAN);

  // The out_k value is set to 0 because the operator handler does not want
  // the output pointer to be involved in the calculation. TODO INT?
  float out_k = 0.0;
  auto alpha = alpha_scalar.to<float>();
  auto beta = beta_scalar.to<float>();

  /* C = op(alpha1 * A, alpha2 * B) + beta * C */
  AT_DISPATCH_MLU_OPTENSOR_TYPES_AND_HALF
  (
      self_contiguous.scalar_type(), "optensor_internal", [&] {
        auto alpha_value = static_cast<scalar_t>(alpha);
        auto beta_value = static_cast<scalar_t>(beta);
        auto out_value = static_cast<scalar_t>(out_k);
        TORCH_CNNL_CHECK(cnnlOpTensor(
            handle, descOpTensor.desc(), &alpha_value, self_desc.desc(),
            self_ptr, &beta_value, other_desc.desc(), other_ptr, temp_ptr,
            workspace_size, &out_value, output_desc.desc(), output_ptr));
      });
  return output_contiguous;
}

/* output = op(input * alpha, other * beta) */
at::Tensor cnnl_optensor_internal(const at::Tensor& input,
                                  const at::Tensor& other,
                                  at::Scalar alpha_scalar,
                                  at::Scalar beta_scalar,
                                  cnnlOpTensorDesc_t op_type) {
  // Determine the type of computation.
  // Operator calculations support only Float and Half.
  at::ScalarType common_type = at::native::result_type(input, other);
  auto compute_type = at::kFloat;
  auto type_iter = find(type_vec.begin(), type_vec.end(), common_type);
  if (type_iter != type_vec.end()) {
    compute_type = common_type;
  }

  at::Tensor input_tensor = convertScalarToTensor(input, input.scalar_type());
  at::Tensor other_tensor = convertScalarToTensor(other, other.scalar_type());

  // Converts the input/output type to match the calculation type.
  at::Tensor input_temp = convertTensorType(input_tensor, compute_type);
  at::Tensor other_temp = convertTensorType(other_tensor, compute_type);

  // get output shape from input shapes.
  std::vector<int64_t> output_shape = at::infer_size(input_temp.sizes(),
                                                     other_temp.sizes());

  std::vector<at::Tensor> tensor_vec = {input_temp, other_temp};
  auto memory_format = infer_tensor_list_contiguous(tensor_vec);
  auto output = at::empty(output_shape,
                          input_temp.options().dtype(compute_type),
                          infer_memory_format(output_shape.size(), memory_format));

  output =
      cnnl_optensor_compute(output, input_temp, other_temp,
                            alpha_scalar, beta_scalar, op_type,
                            common_type);
  // Restore the transitions of the tensor type, and the output type
  // is the common type of the input tensor type.
  auto result = convertTensorType(output, common_type);
  return result;
}

at::Tensor& cnnl_optensor_out_internal(at::Tensor& output,
                                       const at::Tensor& input,
                                       const at::Tensor& other,
                                       at::Scalar alpha_scalar,
                                       at::Scalar beta_scalar,
                                       cnnlOpTensorDesc_t op_type) {
  if (output.numel() == 0) {
      return output;
  }
  auto output_type = output.scalar_type();
  auto compute_type = get_compute_type(input, other, output);

  at::Tensor output_contiguous;
  if (output.data_ptr()) {
      output_contiguous = cnnl_contiguous(output, output.suggest_memory_format());
  } else {
      auto output_tmp = at::native::empty_like(output,
              output.options().dtype(compute_type).device(at::Device::Type::MLU));
      output_contiguous = cnnl_contiguous(output_tmp, output.suggest_memory_format());
  }

  output_contiguous =
      cnnl_optensor_compute(output_contiguous, input, other,
                            alpha_scalar, beta_scalar, op_type,
                            compute_type);
  output_contiguous = convertTensorType(output_contiguous, output_type);

  // not inplace or out, just return, do not copy.
  if (!output.data_ptr()) {
      output = output_contiguous;
      return output;
  }
  // if output_contiguous is not output, then copy output_contiguous to output
  if (!output.is_contiguous(output.suggest_memory_format()) || output_type != compute_type) {
    output.copy_(output_contiguous);
  }

  return output;
}
}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
