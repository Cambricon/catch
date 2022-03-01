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

#include "aten/util/types.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

using vec = std::vector<cnnlDataType_t>;
std::map<vec, cnnlCastDataType_t> cast_map = {
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_HALF}, CNNL_CAST_FLOAT_TO_HALF},
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT32}, CNNL_CAST_FLOAT_TO_INT32},
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT16}, CNNL_CAST_FLOAT_TO_INT16},
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_INT8}, CNNL_CAST_FLOAT_TO_INT8},
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_UINT8}, CNNL_CAST_FLOAT_TO_UINT8},
    {vec{CNNL_DTYPE_FLOAT, CNNL_DTYPE_BOOL}, CNNL_CAST_FLOAT_TO_BOOL},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_FLOAT}, CNNL_CAST_HALF_TO_FLOAT},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_INT32}, CNNL_CAST_HALF_TO_INT32},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_INT16}, CNNL_CAST_HALF_TO_INT16},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_INT8}, CNNL_CAST_HALF_TO_INT8},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_UINT8}, CNNL_CAST_HALF_TO_UINT8},
    {vec{CNNL_DTYPE_HALF, CNNL_DTYPE_BOOL}, CNNL_CAST_HALF_TO_BOOL},
    {vec{CNNL_DTYPE_INT32, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT32_TO_FLOAT},
    {vec{CNNL_DTYPE_INT32, CNNL_DTYPE_HALF}, CNNL_CAST_INT32_TO_HALF},
    {vec{CNNL_DTYPE_INT32, CNNL_DTYPE_INT8}, CNNL_CAST_INT32_TO_INT8},
    {vec{CNNL_DTYPE_INT32, CNNL_DTYPE_INT16}, CNNL_CAST_INT32_TO_INT16},
    {vec{CNNL_DTYPE_INT16, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT16_TO_FLOAT},
    {vec{CNNL_DTYPE_INT16, CNNL_DTYPE_HALF}, CNNL_CAST_INT16_TO_HALF},
    {vec{CNNL_DTYPE_INT16, CNNL_DTYPE_INT32}, CNNL_CAST_INT16_TO_INT32},
    {vec{CNNL_DTYPE_INT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_INT8_TO_FLOAT},
    {vec{CNNL_DTYPE_INT8, CNNL_DTYPE_HALF}, CNNL_CAST_INT8_TO_HALF},
    {vec{CNNL_DTYPE_INT8, CNNL_DTYPE_INT32}, CNNL_CAST_INT8_TO_INT32},
    {vec{CNNL_DTYPE_UINT8, CNNL_DTYPE_FLOAT}, CNNL_CAST_UINT8_TO_FLOAT},
    {vec{CNNL_DTYPE_UINT8, CNNL_DTYPE_HALF}, CNNL_CAST_UINT8_TO_HALF},
    {vec{CNNL_DTYPE_BOOL, CNNL_DTYPE_FLOAT}, CNNL_CAST_BOOL_TO_FLOAT},
    {vec{CNNL_DTYPE_BOOL, CNNL_DTYPE_HALF}, CNNL_CAST_BOOL_TO_HALF},
    {vec{CNNL_DTYPE_BOOL, CNNL_DTYPE_INT32}, CNNL_CAST_BOOL_TO_INT32}};

void cnnl_cast_internal(const at::Tensor& input, at::Tensor& output) {
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  if (input_impl->numel() == 0) {
    return;
  }
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input);
  output_desc.set(output);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  auto queue = getCurrentQueue();
  cnnlDataType_t src_dtype = input_impl->getCnnlType();
  cnnlDataType_t dst_dtype = output_impl->getCnnlType();
  if (src_dtype == dst_dtype) {
    cnrtDataType_t cnrt_type = fromCnnlType2CnrtType(src_dtype);
    int insize = input_impl->numel() * cnrtDataTypeSize(cnrt_type);
    int outsize = output_impl->numel() * cnrtDataTypeSize(cnrt_type);
    auto copy_size = insize <= outsize ? insize : outsize;
    if (input.device().index() == output.device().index()) {
      cnrtMemcpyAsync(output_ptr, input_ptr, copy_size, queue.queue(),
                      CNRT_MEM_TRANS_DIR_DEV2DEV);
    } else { // cnrtNotifier don't support wait on different device.
      queue.synchronize();
      cnrtMemcpy(output_ptr, input_ptr, copy_size, CNRT_MEM_TRANS_DIR_DEV2DEV);
    }
    return;
  }

  // Determine the data conversion type.
  auto iter = cast_map.find({src_dtype, dst_dtype});
  if (iter == cast_map.end()) {
    // if cnnl do not support the cast type, use half to transform
    auto tmp = at::empty_like(input, input.options().dtype(at::ScalarType::Half));
    cnnl_cast_internal(input, tmp);
    cnnl_cast_internal(tmp, output);
    return;
  }
  TORCH_MLU_CHECK(iter != cast_map.end(), "CNNL don't support cast ",
          input.dtype().name(), " data type to ", output.dtype().name(), " data type!!");
  cnnlCastDataType_t cast_type;
  cast_type = cast_map[vec{src_dtype, dst_dtype}];

  TORCH_CNNL_CHECK(cnnlCastDataType(handle, input_desc.desc(), input_ptr,
                                    cast_type, output_desc.desc(), output_ptr));
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
