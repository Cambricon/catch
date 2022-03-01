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
#include <cmath>
#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

static at::Tensor get_output_info(const at::Tensor& input,
                                  int bitwidth,
                                  cnnlDataType_t *data_type,
                                  c10::MemoryFormat memory_format) {
  at::Tensor output;
  auto input_size = input.sizes();

  switch (bitwidth) {
    case 8:
      output =
          at::empty_like(input, input.options().dtype(at::ScalarType::Char));
      *data_type = CNNL_DTYPE_INT8;
      break;
    case 16:
      output =
          at::empty_like(input, input.options().dtype(at::ScalarType::Short));
      *data_type = CNNL_DTYPE_INT16;
      break;
    case 31:
      output =
          at::empty_like(input, input.options().dtype(at::ScalarType::Int));
      *data_type = CNNL_DTYPE_INT31;
      break;
    default:
      LOG(ERROR) << "bitwidth not support";
  }

  return output;
}

at::Tensor cnnl_quantify_offline_internal(const at::Tensor& input,
                                          const int bitwidth,
                                          const int position) {
  TORCH_CHECK(input.dim() >= 0, "dimension not support");
  TORCH_MLU_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                  "quantify only support input float/half");
  cnnlDataType_t data_type;
  auto memory_format = input.suggest_memory_format();
  at::Tensor output = get_output_info(input, bitwidth, &data_type, memory_format);

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  // get current handle
  auto handle = getCurrentHandle();
  auto suggest_layout = suggest_cnnl_layout(output);
  input_desc.set(input, suggest_layout);
  output_desc.set(output, data_type, position, suggest_layout);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  cnnlQuantizeMode_t mode = CNNL_QUANTIZE_POSITION;
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlQuantizeV1(handle, mode, input_desc.desc(), input_ptr,
        output_desc.desc(), output_ptr));
  output_impl->setCnnlType(data_type);
  CNLOG(INFO) << "[operation] : cnnl_quantify_offline";
  return output;
}

at::Tensor cnnl_quantify_offline_internal(const at::Tensor& input,
                                          const int bitwidth,
                                          const at::Tensor& position) {
  TORCH_MLU_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                  "quantify only support input float/half");
  TORCH_CHECK(input.dim() >= 0,
              "input dimension not support for quantify_offline op.");
  cnnlDataType_t data_type;
  auto memory_format = input.suggest_memory_format();
  at::Tensor output = get_output_info(input, bitwidth, &data_type, memory_format);
  // get tensor impl
  auto input_impl = getMluTensorImpl(input);
  auto position_impl = getMluTensorImpl(position);
  auto output_impl = getMluTensorImpl(output);

  // set desc
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input, CNNL_LAYOUT_NHWC);
  output_desc.set(output, data_type, CNNL_LAYOUT_NHWC);

  // get current handle
  auto handle = getCurrentHandle();

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto position_ptr = position_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  cnnlQuantizeMode_t mode = CNNL_QUANTIZE_POSITION;
  TORCH_CNNL_CHECK(cnnlQuantizeV2(
      handle, mode, input_desc.desc(), input_ptr,
      position_ptr, nullptr, nullptr, output_desc.desc(), output_ptr));
  output_impl->setCnnlType(data_type);

  CNLOG(INFO) << "[operation] : cnnl_quantify_online";
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
