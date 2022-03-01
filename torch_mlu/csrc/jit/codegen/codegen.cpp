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

#include <cstdlib>
#include "jit/codegen/codegen.h"
#include "jit/codegen/io.h"
#include "jit/codegen/control_flow.h"
#include "jit/codegen/evalution/eval.h"
#include "jit/codegen/convertion/convert.h"
#include "aten/util/memory_allocator.h"

#include "cndrv_api.h"  // NOLINT

namespace torch_mlu {
namespace jit {
namespace codegen {

// Use ENABLE_MAGICMIND_FUSION_MODE env to judge enabling fusion mode or not.
#define ENABLE_MAGICMIND_FUSION (std::getenv("ENABLE_MAGICMIND_FUSION_MODE") != nullptr && \
  (strcmp(std::getenv("ENABLE_MAGICMIND_FUSION_MODE"), "off") == 0 || \
   strcmp(std::getenv("ENABLE_MAGICMIND_FUSION_MODE"), "OFF") == 0 ||  \
   strcmp(std::getenv("ENABLE_MAGICMIND_FUSION_MODE"), "0") == 0)  \
                       ? false \
                       : true)

void set_builder_config(MagicmindHandle *handle) {
  // When enabling mm fusion mode, the following configuration options should be set.
  if (ENABLE_MAGICMIND_FUSION) {
    // q_bitwitth = 8->qint8 / 16->qint16 / 32->no_quant
    int q_bitwidth = handle->quantized_bitwidth;
    // i_fp_mode = 0->fp16 / 1->fp32
    int i_fp_mode = (handle->has_fp16_input) ? 0 : 1;
    // i_precision =
    // 8  -> qint8_mixed_float16
    // 16 -> qint16_mixed_float16
    // 9  -> qint8_mixed_float32
    // 17 -> qint16_mixed_float32
    // 32 -> force_float16
    // 33 -> force_float32
    int i_precision = q_bitwidth + i_fp_mode;

    handle->builder_config->ParseFromString(
              R"({"graph_shape_mutable": false, "opt_config": {"auto_fusion_enable": true}})");
    handle->builder_config->ParseFromString(R"({"opt_config": {"conv_scale_fold": true}})");

    switch (i_precision) {
      case 8:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "qint8_mixed_float16"}})");
        break;
      case 9:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "qint8_mixed_float32"}})");
        break;
      case 16:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "qint16_mixed_float16"}})");
        break;
      case 17:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "qint16_mixed_float32"}})");
        break;
      case 32:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "force_float16"}})");
        break;
      case 33:
        handle->builder_config->ParseFromString(
                R"({"precision_config": {"precision_mode": "force_float32"}})");
        break;
      default:
        CNLOG(ERROR) <<
          "Incorrect config on precision mode, please check your network script file.";
        break;
    }
  }

  #define MAX_DEVICENAME_LENGTH 100
  char device_name[MAX_DEVICENAME_LENGTH];
  cnDeviceGetName(device_name, MAX_DEVICENAME_LENGTH, handle->device_id);
  #undef MAX_DEVICENAME_LENGTH

  auto mm_supported_arch = handle->builder_config->GetMLUArch();
  bool dev_flags = false;
  for (auto arch : mm_supported_arch) {
    if (std::string(device_name).find(arch) != std::string::npos) {
      std::vector<std::string> arch_vector;
      arch_vector.push_back(arch);
      handle->builder_config->SetMLUArch(arch_vector);
      dev_flags = true;
    }
  }

  if (!dev_flags) {
    TORCH_WARN("Can't match the device name between real device card and ",
               "MagicMind supported device card, so compile all device kernels in default.");
  }
}

MagicmindHandle::MagicmindHandle() {
  network = magicmind_unique_ptr<magicmind::INetwork>(magicmind::CreateINetwork());
  builder = magicmind_unique_ptr<magicmind::IBuilder>(magicmind::CreateIBuilder());
  builder_config =
      magicmind_unique_ptr<magicmind::IBuilderConfig>(magicmind::CreateIBuilderConfig());
}

MagicmindHandle::~MagicmindHandle() {
  for (auto ptr : persistent_buffers) {
    torch_mlu::memory::deallocateMemory<void>(ptr);
  }
}

void MagicmindHandle::bindingValueAndIvalue(
    const torch::jit::Value* value, torch::jit::IValue ivalue) {
  conversion_value_map[value] = std::move(ivalue);
}

magicmind_unique_ptr<magicmind::IModel> convertSubGraphToIModel(
    MagicmindHandle *handle,
    const torch::jit::Block* block,
    const at::ArrayRef<torch::jit::IValue>& stack) {
  TORCH_CHECK(handle != nullptr,
              "handle should not be nullptr for convertSubGraphToIModel().");
  TORCH_CHECK(block != nullptr,
              "block should not be nullptr for convertSubGraphToIModel().");
  // add static parameters to network
  addStaticParameters(handle, block->inputs(), stack);

  // add inputs to network
  if (!addInputs(handle, block->inputs(), stack)) return nullptr;

  auto nodes = block->nodes();
  for (auto node : nodes) {
    if (evalution::isEvalNode(node)) {
      auto eval_out = evalution::evalNode(handle, node);
      if (eval_out) {
        handle->bindingValueAndIvalue(node->output(0), eval_out.value());
      }
    } else if (convertion::isConvertNode(node)) {
      convertion::convertNode(handle, node);
    } else if (node->kind() == torch::jit::prim::Loop) {
      // TODO(kongweiguang)
      convertLoopNode(handle, node);
    } else if (node->kind() == torch::jit::prim::If) {
      // TODO(kongweiguang)
      convertIfNode(handle, node);
    }
  }

  // add outputs to network
  if (!markOutputs(handle, block->outputs())) return nullptr;

  // set the builder config
  set_builder_config(handle);

  // generate the magicmind::IModel
  return magicmind_unique_ptr<magicmind::IModel>(handle->builder->BuildModel("test_model",
                                                 handle->network.get(),
                                                 handle->builder_config.get()));
}

}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
