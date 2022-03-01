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

#include "jit/runtime/graph_runner.h"
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include "jit/utils/utils.h"
#include "interface_runtime.h"  // NOLINT
#include "aten/util/cnlog.h"

namespace torch_mlu {
namespace jit {
namespace runtime {

MMGraphRunner::DebugType MMGraphRunner::get_debug_type() {
  const char* compare_with_jit = getenv("FUSED_KERNEL_DEBUG");
  if (compare_with_jit &&
     (strcmp(compare_with_jit, "CPU") == 0 || strcmp(compare_with_jit, "cpu") == 0)) {
    return DebugType::CPU;
  }
  if (compare_with_jit &&
     (strcmp(compare_with_jit, "CNNL") == 0 || strcmp(compare_with_jit, "cnnl") == 0)) {
    return DebugType::CNNL;
  }
  return DebugType::NONE;
}

// return the current number of current fused kernel; if debug is disabled, return -1;
int MMGraphRunner::debug_fused_op(torch::jit::Stack& stack) {
  auto debug_type = get_debug_type();
  if (debug_type == DebugType::NONE) {
    return -1;
  }
  torch::jit::Stack copyStack;
  std::string debug_tag;
  if (debug_type == DebugType::CPU) {
    debug_tag = "cpu";
    for (auto &ival : stack) {
      if (ival.isTensor()) {
        auto cpu_tensor = ival.deepcopy().toTensor().to("cpu");
        copyStack.push_back(cpu_tensor);
      } else if (ival.isModule()) {
        auto cpu_module = ival.deepcopy().toModule();
        cpu_module.to(at::kCPU);
        copyStack.push_back(cpu_module._ivalue());
      } else {
        CNLOG(DBG) << "stack has non tensor/module ival";
        copyStack.push_back(ival);
      }
    }
  } else if (debug_type == DebugType::CNNL) {
    debug_tag = "catch_cnnl";
    for (auto &ival : stack) {
      if (ival.isTensor()) {
        copyStack.push_back(ival.deepcopy());
      } else {
        CNLOG(DBG) << "stack has non tensor ival";
        copyStack.push_back(ival.deepcopy());
      }
    }
  }
  runFallback(copyStack);
  // record the current number of the fused kernel;
  static int id = 0;
  int idx_out = 0;
  for (auto o : copyStack) {
    if (!o.isTensor()) {
      continue;
    }
    std::string file = debug_tag + "_jitouttensor_MLUFusionGroup" +
                       std::to_string(id)+"_" + std::to_string(idx_out);
    utils::saveTensor(o.toTensor().to("cpu"), file);
    ++idx_out;
  }
  return id++;
}

void MMGraphRunner::runFallback(torch::jit::Stack& stack) {
  torch::jit::InterpreterState{code()}.run(stack);
}

std::shared_ptr<MMFusedKernel> MMGraphRunner::compile_kernel(
    const MMArgSpec& arg_spec,
    const at::ArrayRef<torch::jit::IValue>& all_inputs,
    int dev_id) {
  const std::vector<MMTensorDesc>& input_desc = arg_spec.descs();

  auto graph = graph_->copy();
  int id = 0;
  for (auto input : graph->inputs()) {
    if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
      AT_ASSERT(id < input_desc.size());
      const auto& desc = input_desc[id];
      std::vector<int64_t> sizes{desc.sizes_.begin(), desc.sizes_.end()};
      input->setType(c10::TensorType::createContiguous(
        desc.scalar_type,
        c10::DeviceType::CPU,
        sizes));
      ++id;
    }
  }

  torch::jit::PropagateInputShapes(graph);

  codegen::MagicmindHandle handle;
  handle.device_id = dev_id;
  auto imodel = codegen::convertSubGraphToIModel(&handle, graph->block(), all_inputs);
  if (!imodel) {
    CNLOG(ERROR) << "Convert graph to IModel failed";
    return nullptr;
  }

  CNRT_CHECK(cnrtSetDevice(dev_id));
  magicmind::IModel::EngineConfig config;
  config.device_type = "MLU";
  // When const_data_init is set to true, const data is initialized at CreateIEngine(), but you
  // can't call SetConstData() after calling CreateIEngine(). You can destroy model
  // after calling CreateIEngine()
  config.const_data_init = true;
  auto iengine = codegen::magicmind_unique_ptr<magicmind::IEngine>(imodel->CreateIEngine(config));
  // one IEngine for one device,
  // but one IModel can generate multiple IEngine，
  // different IEngine can run on the same or different devices
  auto icontext = codegen::magicmind_unique_ptr<magicmind::IContext>(iengine->CreateIContext());
  return std::make_shared<MMFusedKernel>(
      std::move(iengine), std::move(icontext), input_desc);
}

std::vector<at::Tensor> MMGraphRunner::launch_fusion(std::shared_ptr<MMFusedKernel> fusion,
  const std::vector<at::Tensor>& inputs, int device_id) {
    return fusion->launch_raw(device_id, inputs);
  }

bool MMGraphRunner::runMLUFusion(torch::jit::Stack& stack) {
  auto all_inputs = torch::jit::last(stack, graph_->inputs().size());

  std::vector<at::Tensor> tensor_inputs;
  // Only support tensor and tensor list as input, other type like Int,IntList are
  // taken care by codegen::addStaticParameters as constant values
  for (size_t i = 0; i < all_inputs.size(); i++) {
    if (all_inputs[i].isTensor()) {
      tensor_inputs.emplace_back(all_inputs[i].toTensor());
    } else if (all_inputs[i].isTensorList()) {
      for (auto &t : all_inputs[i].toTensorVector()) {
        tensor_inputs.emplace_back(t);
      }
    }
  }
  int device_id = 0;
  // check tensor inputs
  if (!tensor_inputs.empty()) {
    if (!tensor_inputs.at(0).defined()) {
      LOG(INFO) << "Undefined input tensor";
      return false;
    }
    // Determines device to dispatch to.
    at::Device device = tensor_inputs.at(0).device();
    device_id = device.index();
    // if input device type is not mlu, fall back to cpu
    if (device.type() != c10::kMLU) {
      LOG(INFO) << "Input tensor is not mlu tensor";
      return false;
    }
    // If there's a device mismatch in the inputs or if one of the input is a
    // sparse tensor, we use the fallback (which should give a nice error
    // message).
    for (const auto& t : at::TensorList(tensor_inputs).slice(1)) {
      // Sparse tensor could not by supported by CUDA fusion, so we bail out.
      if (t.device() != device || t.is_sparse()) {
        LOG(ERROR) << "Not Supported: "
                   << "input tensors have different device or one of them is sparse tensor";
        return false;
      }
    }
  }

  // Retrieves the kernel, compiling (and caching) if necessary
  MMArgSpec arg_spec{tensor_inputs, device_id};
  auto maybe_kernel = find_kernel(arg_spec);
  if (!maybe_kernel) {
    const auto kernel = compile_kernel(arg_spec, all_inputs, device_id);
    cache_kernel(arg_spec, kernel);
  }
  maybe_kernel = find_kernel(arg_spec);
  AT_ASSERT(maybe_kernel);
  auto outtensors = launch_fusion(*maybe_kernel, tensor_inputs, device_id);

  int debug_idx = debug_fused_op(stack);
  torch::jit::drop(stack, all_inputs.size());
  // debug_idx is -1 when debug is disabled
  if (debug_idx != -1) {
    int id = 0;
    for (auto o : outtensors) {
      std::string file = "mmouttensor_MLUFusionGroup" + std::to_string(debug_idx) +
                         "_" + std::to_string(id);
      utils::saveTensor(o.to("cpu"), file);
      ++id;
    }
  }
  stack.insert(stack.end(),
               std::make_move_iterator(outtensors.begin()),
               std::make_move_iterator(outtensors.end()));
  return true;
}

void MMGraphRunner::runFusion(torch::jit::Stack& stack) {
  const auto result = runMLUFusion(stack);
  if (!result) {
    TORCH_WARN("Run mm FusedKernel failed, fallback to catch cnnl");
    runFallback(stack);
  }
}

}  // namespace runtime
}  // namespace jit
}  // namespace torch_mlu


