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
#include "aten/util/exceptions.h"
#include "aten/util/cnlog.h"

namespace torch_mlu {
cnrtDataType_t fromCnnlType2CnrtType(cnnlDataType_t cnnl_data_type) {
  switch (cnnl_data_type) {
    case CNNL_DTYPE_HALF:
        return CNRT_FLOAT16;
    case CNNL_DTYPE_FLOAT:
        return CNRT_FLOAT32;
    case CNNL_DTYPE_INT32:
        return CNRT_INT32;
    case CNNL_DTYPE_INT8:
        return CNRT_INT8;
    case CNNL_DTYPE_UINT8:
        return CNRT_UINT8;
    case CNNL_DTYPE_INT16:
        return CNRT_INT16;
    case CNNL_DTYPE_BOOL:
        return CNRT_BOOL;
    default:
        LOG(ERROR) << "Invalid data type from cnnl to cnrt!";
      return CNRT_INVALID;
  }
}


// transfer tensor datatype into cnnl datatype
cnnlDataType_t getCnnlDataType(const caffe2::TypeMeta& data_type) {
  if (data_type.name() == std::string("float")) {
    return CNNL_DTYPE_FLOAT;
  } else if (data_type.name() == std::string("double")) {
    return CNNL_DTYPE_FLOAT;
  } else if (data_type.name() == std::string("c10::Half")) {
    return CNNL_DTYPE_HALF;
  } else if (data_type.name() == std::string("int")) {
    return CNNL_DTYPE_INT32;
  } else if (data_type.name() == std::string("int8")) {
    return CNNL_DTYPE_INT8;
  } else if (data_type.name() == std::string("unsigned char")) {
    return CNNL_DTYPE_UINT8;
  } else if (data_type.name() == std::string("signed char")) {
    return CNNL_DTYPE_INT8;
  } else if (data_type.name() == std::string("bool")) {
    return CNNL_DTYPE_BOOL;
  } else if (data_type.name() == std::string("short int")) {
    return CNNL_DTYPE_INT16;
  } else if (data_type.name() == std::string("short")) {
    return CNNL_DTYPE_INT16;
    // XXX: Another interface is required for mlu type translation
  } else if (data_type.name() == std::string("long int")) {
    return CNNL_DTYPE_INT32;
  } else if (data_type.name() == std::string("long")) {
    return CNNL_DTYPE_INT32;
  }
  std::string msg("getCnnlDataType() not supported for ");
  msg = msg + data_type.name().data();
  throw std::runtime_error(msg);
}


cnrtDataType_t CnnlType2CnrtType(cnnlDataType_t cnnl_data_type) {
  switch (cnnl_data_type) {
    case CNNL_DTYPE_HALF:
      return CNRT_FLOAT16;
    case CNNL_DTYPE_FLOAT:
      return CNRT_FLOAT32;
    case CNNL_DTYPE_INT32:
      return CNRT_INT32;
    case CNNL_DTYPE_INT8:
      return CNRT_INT8;
    case CNNL_DTYPE_UINT8:
      return CNRT_UINT8;
    case CNNL_DTYPE_INT16:
      return CNRT_INT16;
    case CNNL_DTYPE_INT31:
      return CNRT_INT32;
    case CNNL_DTYPE_BOOL:
      return CNRT_BOOL;
    default: {
      LOG(ERROR) << "Invalid data type from cnnl to cnrt!";
      return CNRT_INVALID;
    }
  }
}


cnrtDataType_t toCnrtDtype(const caffe2::TypeMeta& data_type) {
  if (data_type.name() == std::string("float")) {
    return CNRT_FLOAT32;
  } else if (data_type.name() == std::string("double")) {
    return CNRT_FLOAT64;
  } else if (data_type.name() == std::string("c10::Half")) {
    return CNRT_FLOAT16;
  } else if (data_type.name() == std::string("int")) {
    return CNRT_INT32;
  } else if (data_type.name() == std::string("int8")) {
    return CNRT_INT8;
  } else if (data_type.name() == std::string("bool")) {
    return CNRT_BOOL;
  } else if (data_type.name() == std::string("long")) {
    return CNRT_INT64;
  } else if (data_type.name() == std::string("long int")) {
    return CNRT_INT64;
  } else if (data_type.name() == std::string("short")) {
    return CNRT_INT16;
  } else if (data_type.name() == std::string("short int")) {
    return CNRT_INT16;
  } else if (data_type.name() == std::string("unsigned char")) {
    return CNRT_UINT8;
  } else if (data_type.name() == std::string("signed char")) {
    return CNRT_INT8;
  } else {
    std::string msg("to_cnrt_dtype: not supported for ");
    msg = msg + data_type.name().data();
    LOG(ERROR) << msg;
    return CNRT_INVALID;
  }
}

// transfer cnrt datatype back to tensor ScalarType datatype
at::ScalarType cnrtType2ScalarType(cnrtDataType_t cnrt_dtype) {
  switch (cnrt_dtype) {
    case CNRT_FLOAT32:
      return at::ScalarType::Float;
    case CNRT_FLOAT64:
      return at::ScalarType::Double;
    case CNRT_FLOAT16:
      return at::ScalarType::Half;
    case CNRT_INT64:
      return at::ScalarType::Long;
    case CNRT_INT32:
      return at::ScalarType::Int;
    case CNRT_INT8:
      return at::ScalarType::Char;
    case CNRT_UINT8:
      return at::ScalarType::Byte;
    case CNRT_BOOL:
      return at::ScalarType::Bool;
    case CNRT_INT16:
      return at::ScalarType::Short;
    default: {
      CNLOG(ERROR) << "Unsupported data type from cnrtDataType_t to at::ScalarType!";
      return at::ScalarType::Undefined;
    }
  }
}

}  // namespace torch_mlu
