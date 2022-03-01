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

#include "jit/utils/data_type.h"

namespace torch_mlu {
namespace jit {
namespace utils {

magicmind::DataType scalarTypeToMagicmindDataType(c10::ScalarType scalar_type) {
  magicmind::DataType mm_data_type;

  switch (scalar_type) {
    case at::kFloat:
      mm_data_type = magicmind::DataType::FLOAT32;
      break;
    case at::kHalf:
      mm_data_type = magicmind::DataType::FLOAT16;
      break;
    case at::kLong:
      mm_data_type = magicmind::DataType::INT64;
      break;
    case at::kInt:
      mm_data_type = magicmind::DataType::INT32;
      break;
    case at::kShort:
      mm_data_type = magicmind::DataType::INT16;
      break;
    case at::kByte:
      mm_data_type = magicmind::DataType::UINT8;
      break;
    case at::kChar:
      mm_data_type = magicmind::DataType::INT8;
      break;
    case at::kBool:
      mm_data_type = magicmind::DataType::BOOL;
      break;
    default:
      AT_ERROR("Unsupported scalar type of pytorch tensor.");
      break;
  }

  return mm_data_type;
}

c10::ScalarType magicmindDataTypeToScalarType(magicmind::DataType data_type) {
  at::ScalarType scalar_type;

  switch (data_type) {
    case magicmind::DataType::FLOAT32:
      scalar_type = at::kFloat;
      break;
    case magicmind::DataType::FLOAT16:
      scalar_type = at::kHalf;
      break;
    case magicmind::DataType::INT64:
      scalar_type = at::kLong;
      break;
    case magicmind::DataType::INT32:
      scalar_type = at::kInt;
      break;
    case magicmind::DataType::INT16:
      scalar_type = at::kShort;
      break;
    case magicmind::DataType::INT8:
      scalar_type = at::kChar;
      break;
    case magicmind::DataType::UINT8:
      scalar_type = at::kByte;
      break;
    case magicmind::DataType::BOOL:
      scalar_type = at::kBool;
      break;
    default:
      AT_ERROR("Unsupported data type of magicmind tensor.");
      break;
  }

  return scalar_type;
}

}  // namespace utils
}  // namespace jit
}  // namespace torch_mlu
