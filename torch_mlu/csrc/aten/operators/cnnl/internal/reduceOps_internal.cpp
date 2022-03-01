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

#include <vector>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

using at::kBool;
using at::kByte;
using at::kChar;
using at::kShort;
using at::kInt;
using at::kLong;
using at::kHalf;
using at::kFloat;
using at::kDouble;

std::map<cnnlReduceOp_t, std::vector<at::ScalarType>> supported_type_table = {
    {CNNL_REDUCE_ADD  , {kHalf, kFloat}},
    {CNNL_REDUCE_AVG  , {kHalf, kFloat}},
    {CNNL_REDUCE_MUL  , {kInt, kHalf, kFloat}},
    {CNNL_REDUCE_MAX  , {kInt, kHalf, kFloat}},
    {CNNL_REDUCE_MIN  , {kInt, kHalf, kFloat}},
    {CNNL_REDUCE_AND  , {kBool, kByte, kChar, kHalf, kFloat}},
    {CNNL_REDUCE_OR   , {kBool, kByte, kChar, kHalf, kFloat}},
    {CNNL_REDUCE_NORM1, {kHalf, kFloat}},
    {CNNL_REDUCE_NORM2, {kHalf, kFloat}}
};

namespace {

inline at::ScalarType find_supported_type(cnnlReduceOp_t mode, at::ScalarType input_type) {
  auto table = supported_type_table.find(mode);
  TORCH_MLU_CHECK(table != supported_type_table.end(),
                 "An unknown reduceOp mode : ", mode, " called.");
  auto type = find(table->second.begin(), table->second.end(), input_type);
  if (type == table->second.end()) {
    CNLOG(INFO) << "ReduceOp input dtype is not supported, will cast to float for caculation."
                << std::endl;
    return kFloat;
  }
  return *type;
}

inline cnnlReduceOp_t getReduceOp(ReduceType reduce_type) {
  switch (reduce_type) {
    case ReduceType::Reduce_Sum:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_ADD" << std::endl;
      return CNNL_REDUCE_ADD;
    case ReduceType::Reduce_Mean:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_AVG" << std::endl;
      return CNNL_REDUCE_AVG;
    case ReduceType::Reduce_Max:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_MAX" << std::endl;
      return CNNL_REDUCE_MAX;
    case ReduceType::Reduce_Min:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_MIN" << std::endl;
      return CNNL_REDUCE_MIN;
    case ReduceType::Reduce_Any:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_OR (ANY)" << std::endl;
      return CNNL_REDUCE_OR;
    case ReduceType::Reduce_And:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_AND" << std::endl;
      return CNNL_REDUCE_AND;
    case ReduceType::Reduce_All:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_AND (ALL)" << std::endl;
      return CNNL_REDUCE_AND;
    case ReduceType::Reduce_Mul:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_MUL" << std::endl;
      return CNNL_REDUCE_MUL;
    case ReduceType::Reduce_Norm1:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_NORM1" << std::endl;
      return CNNL_REDUCE_NORM1;
    case ReduceType::Reduce_Norm2:
      CNLOG(INFO) << "[ReduceMode]: CNNL_REDUCE_NORM2" << std::endl;
      return CNNL_REDUCE_NORM2;
    default:
      return CNNL_REDUCE_ADD;
  }
}

}  // namespace

void cnnl_reduce_internal(const at::Tensor& input, at::Tensor& output,
                          at::Tensor& index, const std::vector<int64_t> reduce_dim,
                          const std::vector<int64_t> desc_shape, ReduceType reduce_type,
                          const std::vector<int64_t> output_shape) {
  /*
    cnnlReduceOps does not squeeze shape, no matter if keepdim is enabled or not.
    So desc_shpae is the same length as input.dim() with only reduced axis is 1,
    and output_shape is the expect shape of output due to keepdim.
  */

  if (input.numel() == 0) {
    switch (reduce_type) {
      case ReduceType::Reduce_Mean:
        output = at::zeros(output_shape, input.options()) /
                 at::zeros(output_shape, input.options());  // 0/0 ->nan
        return;
      case ReduceType::Reduce_Mul:
        output = at::ones(output_shape, input.options());
        return;
    }
    output = at::zeros(output_shape, input.options());
    return;
  }
  // Only Min and Max Ops have indices as result.
  auto reduce_indices = (reduce_type == ReduceType::Reduce_Min
                         || reduce_type == ReduceType::Reduce_Max) ?
                        CNNL_REDUCE_FLATTENED_INDICES : CNNL_REDUCE_NO_INDICES;
  auto reduce_indices_type = CNNL_32BIT_INDICES;
  auto reduce_mode = getReduceOp(reduce_type);
  auto reduce_dtype = find_supported_type(reduce_mode, input.scalar_type());

  auto input_t = input;
  if (input.scalar_type() != reduce_dtype) {
    input_t = input.to(reduce_dtype);
  }

  output = at::empty(output_shape, input_t.options(), c10::MemoryFormat::Contiguous);

  CnnlReduceDescriptor reduce_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;

  auto input_impl = getMluTensorImpl(input_t);
  auto output_impl = getMluTensorImpl(output);

  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  void * index_ptr = nullptr;

  if (reduce_indices != CNNL_REDUCE_NO_INDICES) {
    index = at::empty(output_shape, input_t.options().dtype(caffe2::TypeMeta::Make<long>()),
                      c10::MemoryFormat::Contiguous);
    auto index_impl = getMluTensorImpl(index);
    index_ptr = index_impl->cnnlMalloc();
  } else {
    index = at::empty({0}, input_t.options().dtype(caffe2::TypeMeta::Make<long>()));
  }

  if (reduce_dim.size() == input.dim()) {
    /*
      FULL-REDUCE:
      If ReduceOp select all dims and reduce them to a 1-item scalar tensor,
      use full-reduce mode which has axis as [-1] instead of [0, 1, 2, ..., n].
    */
    std::vector<int64_t> full_reduce(1, -1);
    std::vector<int64_t> fake_size(input_t.dim(), 1);
    reduce_desc.set(input_t, full_reduce, reduce_mode, reduce_indices, reduce_indices_type);
    input_desc.set_reduce(input_t);
    at::Tensor fake_tensor = at::empty(fake_size, output.options());
    output_desc.set_reduce(fake_tensor);
    at::Tensor fake_tensor2 = at::empty(fake_size, index.options());
    index_desc.set_reduce(fake_tensor2);
  } else {
    reduce_desc.set(input_t, reduce_dim, reduce_mode, reduce_indices, reduce_indices_type);
    input_desc.set_reduce(input_t);
    output_desc.set_reduce(output, desc_shape);
    index_desc.set_reduce(index, desc_shape);
  }
  uint32_t index_size_inbytes = sizeof(int) * index.numel();

  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  at::Tensor workspace;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(
      handle, input_desc.desc(), output_desc.desc(), reduce_desc.mut_desc(),
      &workspace_size));
  if (workspace_size != 0) {
    workspace =
        at::empty(workspace_size, input.options().dtype(at::ScalarType::Char));
    workspace_ptr = getMluTensorImpl(workspace)->cnnlMalloc();
  }

  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlReduce(
      /* handle               */ handle,
      /* reduce_desc          */ reduce_desc.desc(),
      /* workspace            */ workspace_ptr,
      /* workspace_size       */ workspace_size,
      /* alpha                */ alpha,
      /* input_desc           */ input_desc.desc(),
      /* input                */ input_ptr,
      /* indices_size_inbytes */ index_size_inbytes,
      /* indices              */ index_ptr,
      /* beta                 */ beta,
      /* output_desc          */ output_desc.desc(),
      /* output               */ output_ptr));

  index = index.to(kLong);
  if (reduce_type == ReduceType::Reduce_Sum) {
    if (input.scalar_type() != kFloat &&
        input.scalar_type() != kDouble &&
        input.scalar_type() != kHalf) {
      // Sum Op always return int64(kLong) for non-floating type.
      output = output.to(kLong);
      return;
    }
  }
  if (output.scalar_type() != input.scalar_type()) {
    output = output.to(input.scalar_type());
  }
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

