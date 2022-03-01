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

#include <ATen/Context.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/CPUGeneratorImpl.h>
#include <torch/library.h>

#include "aten/core/tensor_impl.h"
#include "aten/generated/aten_mlu_type_default.h"
#include "aten/generated/aten_mlu_type.h"
#include "aten/util/cnlog.h"
#include "aten/util/tensor_util.h"

namespace torch_mlu {


void inputConvertAndPushTensor(torch::jit::Stack& stack, at::Tensor&& tensor, int direction,
                              InplaceOpArgsRegistry& reg, int index, bool isMutable) {
  if (!tensor.defined()) {  // undefined tensors do not go through registration
      torch::jit::push(stack, std::move(tensor));  // push to the stack and give up control
      return;
  }
  if (direction == MLU2CPU) {  // mlu --> cpu
    auto src_impl = getMluTensorImpl(tensor);
    at::Tensor dst_tensor;
    if (src_impl->is_empty()) {
      // currently tensor.cpu() cannot handle empty tensor copy,
      // so we resort to the empty_like()
      dst_tensor = at::empty_like(tensor, tensor.options().device(at::kCPU));
    } else {
      dst_tensor = tensor.cpu();  // it is the cpu tensor that got registered!
    }
    // after push, the registery
    // and the stack share the ownership of the tensor
    reg.registerTensor(dst_tensor, index, isMutable);
    torch::jit::push(stack, std::move(dst_tensor));
  }
}


void inputConvertAndPushTensor(c10::List<at::Tensor>& tensorList, at::Tensor&& tensor,
                int direction, InplaceOpArgsRegistry& reg, int index, bool isMutable) {
  if (!tensor.defined()) {  // undefined tensors do not go through registration
      tensorList.push_back(std::move(tensor));   // push to the tensorList and give up control
      return;
  }
  if (direction == MLU2CPU) {
    auto src_impl = getMluTensorImpl(tensor);
    at::Tensor dst_tensor;
    if (src_impl->is_empty()) {
      // currently tensor.cpu() cannot handle empty tensor copy,
      // so we resort to the empty_like()
      dst_tensor = at::empty_like(tensor, tensor.options().device(at::kCPU));
    } else {
      dst_tensor = tensor.cpu();  // it is the cpu tensor that got registered!
    }
    // tensorList.size() is the next available position
    reg.registerTensor(dst_tensor, index, tensorList.size(), isMutable);
    // after push, the registery and the stack share the ownership of the tensor
    tensorList.push_back(std::move(dst_tensor));
  }
}

void outputConvertAndPushTensor(torch::jit::Stack& stack, at::Tensor&& tensor, int direction,
                              InplaceOpArgsRegistry& reg) {
  if (!tensor.defined()) {  // undefined tensors do not go through registration
    // push to the stack and give up control
    torch::jit::push(stack, std::move(tensor));
    return;
  }
  if (direction != CPU2MLU) {
    return;
  }
  // cpu --> mlu
  auto new_tensor = tensor.to("mlu");
  c10::optional<std::pair<at::Tensor, bool>> optionalTensorMutablePair =
                                                    reg.getOriginalMLUTensor(tensor);
  // it is the cpu tensor that is used as a key
  if (!optionalTensorMutablePair) {  // not found, means not an inplace or out argument
    torch::jit::push(stack, std::move(new_tensor));
    return;
  }
  if (!optionalTensorMutablePair.value().second) {  // const tensor
      // push the original, no move
    torch::jit::push(stack, optionalTensorMutablePair.value().first);
    return;
  }

  // mutable tensor
  // inplace-copy the result to the original
  at::Tensor &originalTensor = optionalTensorMutablePair.value().first;
  if (originalTensor.storage().nbytes() == new_tensor.storage().nbytes()) {
    optionalTensorMutablePair.value().first.copy_(new_tensor);
  } else {
    // we will do something like
    // originalTensor.unsafeGetTensorImpl()->shallow_copy_from(...)
    originalTensor.storage().set_nbytes(new_tensor.storage().nbytes());
    // the storage got via originalTensor.storage() is a const storage
    // to make the storage mutable, we use unsafeGetStorageImpl()
    // to achieve const-correctness
    originalTensor.storage().unsafeGetStorageImpl()->set_data_ptr(
            at::DataPtr(new_tensor.storage().data(), new_tensor.storage().device()));
    // no need to call originalTensor.unsafeGetTensorImpl()->bump_version();
    // because autograd framework will take care of it, like in
    // VariableType_2.cpp:bernoulli__float(...)
  }
  originalTensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          new_tensor.sizes(), new_tensor.strides());
  // push the original
  torch::jit::push(stack, std::move(optionalTensorMutablePair.value().first));
}


void outputConvertAndPushTensor(c10::List<at::Tensor>& tensorList, at::Tensor&& tensor,
                              int direction, InplaceOpArgsRegistry& reg) {
  if (!tensor.defined()) {  // undefined tensors do not go through registration
    // push to the tensorList and give up control
    tensorList.push_back(std::move(tensor));
    return;
  }
  if (direction != CPU2MLU) {
    return;
  }
  // cpu --> mlu
  auto new_tensor = tensor.to("mlu");
  c10::optional<std::pair<at::Tensor, bool>> optionalTensorMutablePair =
                                              reg.getOriginalMLUTensor(tensor);
  if (!optionalTensorMutablePair) {  // not found, means not an inplace or out argument
    tensorList.push_back(std::move(new_tensor));
    return;
  }
  // found, means an inplace or out argument
  if (!optionalTensorMutablePair.value().second) {  // const tensor
    // push the original, no move
    tensorList.push_back(optionalTensorMutablePair.value().first);
    return;
  }
  // mutable tensor
  // inplace-copy the result to the original
  at::Tensor &originalTensor = optionalTensorMutablePair.value().first;
  if (originalTensor.storage().nbytes() == new_tensor.storage().nbytes()) {
    optionalTensorMutablePair.value().first.copy_(new_tensor);
  } else {
    // we will do something like
    // originalTensor.unsafeGetTensorImpl()->shallow_copy_from(...)
    originalTensor.storage().set_nbytes(new_tensor.storage().nbytes());
    // the storage got via originalTensor.storage() is a const storage
    // to make the storage mutable, we use unsafeGetStorageImpl() to
    // achieve const-correctness
    originalTensor.storage().unsafeGetStorageImpl()->set_data_ptr(
            at::DataPtr(new_tensor.storage().data(), new_tensor.storage().device()));
    // no need to call originalTensor.unsafeGetTensorImpl()->bump_version();
    // because autograd framework will take care of it, like in
    // VariableType_2.cpp:bernoulli__float(...)
  }
  originalTensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          new_tensor.sizes(), new_tensor.strides());
  tensorList.push_back(std::move(originalTensor));
}

bool is_op_seen(const std::string& opName) {
  static std::unordered_set<std::string> seen {};
  auto p = seen.insert(opName);
  return !p.second;
}

// according to the design of torch::jit::Stack:
// ***operations are defined in a way that ownership of inputs can be
// transferred to the operation***, to verify this behavior, you can refer
// to aten/src/ATen/core/stack.h for the definitions of
// static inline IValue pop(Stack& stack) and
// static inline std::vector<IValue> pop(Stack& stack, size_t n)

// since we now have taken the ownership of the inputs, it is at our disposal of
// moving the elements around, and freeing them when we think it necessary.
// the exception is with the inplace and out parameters, for which we MUST NOT
// keep them, and when we finshed the boxed-call, we have to copy the results to
// them and push them back to the stack at right positions.
void mlu_wrapper(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  if (!is_op_seen(op.operator_name().name)) {
    CNLOG(INFO) << "fallback to CPU: operator_name()=" << op.operator_name()
            << ", " << op.debug() <<"\n\n";
  }
  auto args = torch::jit::pop(*stack, num_arguments);  // ownership transferred to args
  InplaceOpArgsRegistry reg(args);
  for (size_t i = 0; i < num_arguments; i++) {
    if (args[i].isTensor()) {  // the replication of Tensors occurs
      // a. the ownership of the original Tensor remains with args,
      // b. the ownership of the converted Tensor is with the stack and the registry
      const auto& aliasInfo = op.schema().arguments()[i].alias_info();
      bool isMutable = aliasInfo && aliasInfo.value().isWrite();  // is the tensor mutable
        inputConvertAndPushTensor(*stack, args[i].toTensor(), MLU2CPU, reg, i, isMutable);
    } else if (args[i].isTensorList()) {  // the replication of TensorList occurs here
      // a. the ownership of the original tensors in TensorList remains with args,
      // b. the ownership of the converted tensors is with the
      //    new-TensorList-in-stack and the registry
      const auto& aliasInfo = op.schema().arguments()[i].alias_info();
      bool isMutable = aliasInfo && aliasInfo.value().isWrite();  // is the tensor mutable
      c10::List<at::Tensor> cpu_inputs;
      for (at::Tensor&& inputElement : args[i].toTensorList()) {
        inputConvertAndPushTensor(cpu_inputs, std::move(inputElement),
                                                MLU2CPU, reg, i, isMutable);
      }
      torch::jit::push(*stack, std::move(cpu_inputs));
    } else if (args[i].isDevice()) {
      c10::Device originalDevice = args[i].toDevice();
      if (originalDevice.type() == c10::DeviceType::MLU) {
        torch::jit::push(*stack, c10::Device(at::kCPU));
      } else {
        torch::jit::push(*stack, std::move(args[i]));
      }
    } else {  // ownership of elements that are neither Tensor or TensorList is
              // transferred back from args to the stack
      torch::jit::push(*stack, std::move(args[i]));
    }
  }

  op.callBoxed(stack);

  // now the results are already placed in the stack, each in its right position
  // except that they are in the cpu storage.
  // you can look at generated_unboxing_wrappers_2.cpp aten::bernoulli_.float
  // for an example of how this is done in the first place

  // in the ensuing post-processing, we will:
  // a. take them out,
  // b. transfer them to mlu if necessary
  // c. if any of them are registered as an inplace or out parameter,
  //    find the original arg[i], and copy the corresonding result to arg[i]
  // d. place the converted tensors back at the positions from where they was taken out at step a.

  // Rewrap outputs
  auto rets = torch::jit::pop(*stack, num_returns);  // owner ship of cpu tensors to rets
  for (size_t i = 0; i < num_returns; i++) {
    if (rets[i].isTensor()) {  // the replication of Tensors occurs
      // a. the ownership of the original Tensor remains with rets,
      // b. for the converted Tensor
      //    i) if it maps to an inplace or out argument, then a corresponding
      //       registered input argument inherits its content by copying and goes into the stack
      //       (note that it is not shared by the registry because it is an mlu tensor,
      //        only cpu tensors are registered with the InplaceOpArgsRegistry!),
      //       and this first mlu tensor demises after convertAndPushbackTensor.
      //    ii) if it does not map to an inplace or out argument, then it moves from
      //        rets to the stack, and is soly owned by the latter
      outputConvertAndPushTensor(*stack, rets[i].toTensor(), CPU2MLU, reg);
    } else if (rets[i].isTensorList()) {
      c10::List<at::Tensor> mlu_outputs;
      for (at::Tensor&& outputElement : rets[i].toTensorList()) {
        // the replication of Tensors occurs
        // a. the ownership of the original Tensor remains with the TensorList at rets[i],
        // b. for the converted Tensor
        //    i) if it maps to an inplace or out argument, then a corresponding
        //       registered input argument inherits its content by copying and goes into
        //       mlu_outputs, which in turn goes to stack,
        //       (note that it is not shared by the registry because it is an mlu tensor,
        //        only cpu tensors are registered with the InplaceOpArgsRegistry!),
        //        and this first mlu tensor demises after convertAndPushbackTensor.
        //    ii) if it does not map to an inplace or out argument, then it moves
        //        from rets to the stack, as it were in the first place and is
        //        soly owned by the latter
        outputConvertAndPushTensor(mlu_outputs, std::move(outputElement), CPU2MLU, reg);
      }
      torch::jit::push(*stack, std::move(mlu_outputs));
    } else {  // arguments other than Tensor or TensorList were put back to the stack as they were.
      torch::jit::push(*stack, std::move(rets[i]));
    }
  }
}

void RegisterAtenOperators() {
  static auto dispatch = torch::RegisterOperators()
  ${unboxed_only_wrapper_registrations}
   .op("torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
                            torch::RegisterOperators::options()
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<decltype(AtenMluCustomType::nms),
      &AtenMluCustomType::nms>(c10::DispatchKey::MLU))
  ${custom_wrapper_registrations};
}


TORCH_LIBRARY_IMPL(_, MLU, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&mlu_wrapper>());
}

}  // namespace torch_mlu

