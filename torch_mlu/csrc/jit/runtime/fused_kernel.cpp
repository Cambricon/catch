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

#include "jit/runtime/fused_kernel.h"
#include "jit/utils/utils.h"
#include "jit/utils/data_type.h"
#include "aten/cnnl/cnnl_util.h"
#include "aten/util/cnlog.h"
#include "aten/util/tensor_util.h"
#include "cnrt.h"  //  NOLINT

namespace torch_mlu {
namespace jit {
namespace runtime {

std::vector<at::Tensor> MMFusedKernel::launch_raw(int device_id,
    const std::vector<at::Tensor>& inputs) {
  torch_mlu::mlu::MLUGuard guard(device_id);
  cnrtQueue_t queue = getCurQueue();
  std::vector<magicmind::Dims> input_dims;
  for_each(input_desc_.begin(), input_desc_.end(),
          [&input_dims](MMTensorDesc d){input_dims.emplace_back(d.sizes_);});

  // create and get irttensor from context
  std::vector<magicmind::IRTTensor *> input_tensors;
  std::vector<magicmind::IRTTensor *> output_tensors;
  MM_CHECK(CreateInputTensors(icontext_.get(), &input_tensors));
  MM_CHECK(CreateOutputTensors(icontext_.get(), &output_tensors));

  for (uint32_t i = 0; i < input_tensors.size(); ++i) {
    MM_CHECK(input_tensors[i]->SetDimensions(input_dims[i]));
  }

  std::vector<at::Tensor> input_contigous;
  for (auto id = 0; id < inputs.size(); ++id) {
    TORCH_CHECK(inputs[id].device().type() == c10::kMLU);
    input_contigous.emplace_back(cnnl::ops::cnnl_contiguous(inputs[id]));
    MM_CHECK(input_tensors[id]->SetData(input_contigous[id].data_ptr()));
  }

  std::vector<at::Tensor> outtensors;
  auto get_new_mlu_tensorimpl = [&outtensors, &device_id](magicmind::IRTTensor * mmtensor){
    auto dim = mmtensor->GetDimensions().GetDims();
    auto element_type = utils::magicmindDataTypeToScalarType(mmtensor->GetDataType());
    // output tensor format of mm kernel is fixed nchw
    auto options = torch::TensorOptions().dtype(element_type).device(torch::kMLU, device_id)
                                         .memory_format(c10::MemoryFormat::Contiguous);
    auto output = at::zeros(dim, options);
    auto output_impl = getMluTensorImpl(output);
    outtensors.push_back(output);
    return output_impl;
  };
  bool has_mm_cpu_op_output = false;
  for (const auto& output_mm_tensor : output_tensors) {
    if (output_mm_tensor->GetMemoryLocation() == magicmind::TensorLocation::kHost) {
      has_mm_cpu_op_output = true;
      break;
    }
  }
  if (has_mm_cpu_op_output ||
      magicmind::Status::OK() != icontext_->InferOutputShape(input_tensors, output_tensors)) {
    MM_CHECK(icontext_->Enqueue(input_tensors, &output_tensors, queue));
    for (int i = 0; i < output_tensors.size(); ++i) {
      auto output_impl = get_new_mlu_tensorimpl(output_tensors[i]);
      if (output_tensors[i]->GetMemoryLocation() == magicmind::TensorLocation::kMLU) {
        CNRT_CHECK(cnrtMemcpyAsync(output_impl->data(), output_tensors[i]->GetMutableData(),
                                 output_tensors[i]->GetSize(), queue,
                                 CNRT_MEM_TRANS_DIR_DEV2DEV));
      } else {
        CNRT_CHECK(cnrtMemcpyAsync(output_impl->data(), output_tensors[i]->GetMutableData(),
                                   output_tensors[i]->GetSize(), queue,
                                   CNRT_MEM_TRANS_DIR_HOST2DEV));
      }
    }
  } else {
    for (int i = 0; i < output_tensors.size(); ++i) {
      auto output_impl = get_new_mlu_tensorimpl(output_tensors[i]);
      MM_CHECK(output_tensors[i]->SetData(output_impl->data()));
    }
    MM_CHECK(icontext_->Enqueue(input_tensors, output_tensors, queue));
  }
  return outtensors;
}

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu
