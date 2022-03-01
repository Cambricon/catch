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

#include "jit/codegen/tensor.h"
#include "aten/util/memory_allocator.h"

namespace torch_mlu {
namespace jit {
namespace codegen {

static auto itensor_container =
    torch::class_<ITensorContainer>("_catch_jit_ivalue_types",
                                    "ITensorContainer").def(torch::init<>());

bool isITensor(const torch::jit::IValue& ivalue) {
  if (ivalue.isCustomClass()) {
    return true;
  } else {
    return false;
  }
}

magicmind::ITensor* getITensor(const torch::jit::IValue& ivalue) {
  if (isITensor(ivalue)) {
    auto container = ivalue.toCustomClass<ITensorContainer>();
    return container->tensor();
  } else {
    return nullptr;
  }
}

std::vector<magicmind::ITensor*> getITensorVector(const torch::jit::IValue& ivalue) {
  if (isITensor(ivalue)) {
    auto container = ivalue.toCustomClass<ITensorContainer>();
    TORCH_CHECK(container->isITensorVector());
    return container->tensor_vector();
  } else {
    return {};
  }
}

magicmind::ITensor* getOrCreateITensor(MagicmindHandle *handle,
                                       const torch::jit::IValue& ivalue) {
  TORCH_CHECK(handle != nullptr, "handle should not be nullptr for getOrCreateITensor().");
  TORCH_CHECK(isITensor(ivalue) || ivalue.isTensor() || ivalue.isScalar() || ivalue.isIntList(),
              "The IValue type should be CustomClass type or Tensor or Scalar type or IntList.");

  if (isITensor(ivalue)) {
    return getITensor(ivalue);
  } else {
    at::Tensor tensor;
    if (ivalue.isTensor()) {
      tensor = ivalue.toTensor();
    } else if (ivalue.isScalar()) {
      tensor = c10::scalar_to_tensor(ivalue.toScalar());
    } else if (ivalue.isIntList()) {
      auto vec = ivalue.toIntVector();
      int64_t elem_n = vec.size();
      tensor = at::ones({elem_n}, at::kLong).to(at::kCPU);
      for (int k = 0; k < elem_n; k++) {
        static_cast<long*>(tensor.data_ptr())[k] = vec[k];
      }
    }
    return createConstITensor(handle, tensor);
  }
}

magicmind::ITensor* createConstITensor(MagicmindHandle *handle,
                                       at::Tensor tensor) {
  TORCH_CHECK(handle != nullptr, "handle should not be nullptr for createConstITensor().");
  if (tensor.scalar_type() == at::kLong ||
      tensor.scalar_type() == at::kDouble) {
      CNLOG(DBG) << "long/double type will be casted to int32/float type.";
  }

  at::Tensor cast_tensor = tensor.cpu().contiguous();
  at::Tensor const_tensor;

  if (tensor.scalar_type() == at::kLong) {
    const_tensor = cast_tensor.toType(at::kInt);
  } else if (tensor.scalar_type() == at::kDouble) {
    const_tensor = cast_tensor.toType(at::kFloat);
  } else {
    const_tensor = cast_tensor;
  }

  auto mm_data_type = utils::scalarTypeToMagicmindDataType(const_tensor.scalar_type());
  void *temp_buf = nullptr;
  int64_t buf_size;

  switch (mm_data_type) {
    case magicmind::DataType::FLOAT32:
      buf_size = const_tensor.numel() * sizeof(float);
      break;
    case magicmind::DataType::FLOAT16:
      buf_size = const_tensor.numel() * (sizeof(float) / 2);
      break;
    case magicmind::DataType::INT32:
      buf_size = const_tensor.numel() * sizeof(int);
      break;
    case magicmind::DataType::INT8:
      buf_size = const_tensor.numel() * sizeof(char);
      break;
    case magicmind::DataType::BOOL:
      buf_size = const_tensor.numel() * sizeof(bool);
      break;
    default:
      TORCH_CHECK("unsupported magicmind data type.");
      break;
  }

  temp_buf = torch_mlu::memory::allocateMemory<void>(buf_size);
  memcpy(temp_buf, const_tensor.data_ptr(), buf_size);
  handle->persistent_buffers.push_back(temp_buf);

  magicmind::Dims mm_shape = magicmind::Dims(const_tensor.sizes().vec());
  auto const_node = handle->network->AddIConstNode(mm_data_type, mm_shape, temp_buf);
  auto const_itensor = const_node->GetOutput(0);

  return const_itensor;
}

torch::jit::IValue bindITensor(magicmind::ITensor* tensor) {
  TORCH_CHECK(tensor != nullptr, "ITensor should not be nullptr for bindITensor().");

  auto tensor_container = ITensorContainer();
  tensor_container.set_tensor(tensor);

  auto ivalue = c10::IValue(std::move(c10::make_intrusive<ITensorContainer>(tensor_container)));

  return std::move(ivalue);
}

at::Tensor getEmptyTensor(magicmind::ITensor* itensor) {
  auto scalar_type = utils::magicmindDataTypeToScalarType(itensor->GetDataType());
  auto dims = itensor->GetDimension().GetDims();
  for (auto &dim : dims) {
    if (dim == -1) {
      dim = 1;
    }
  }
  auto tensor = at::empty(dims, scalar_type);
  return tensor;
}

torch::jit::IValue bindITensorVector(const std::vector<magicmind::ITensor*>& tensor_vec) {
  TORCH_CHECK(!tensor_vec.empty(), "ITensor vector should not be empty for bindITensorVector().");
  auto tensor_container = ITensorContainer();
  tensor_container.set_tensor_vector(tensor_vec);
  auto ivalue = c10::IValue(std::move(c10::make_intrusive<ITensorContainer>(tensor_container)));
  return ivalue;
}

}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
