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

#include "jit/codegen/convertion/convert.h"

namespace torch_mlu {
namespace jit {
namespace codegen {
namespace convertion {

static auto registry = Registerer()
    .op(R"SIG(aten::max_pool2d(Tensor self, int[2] kernel_size,
                  int[2] stride=[], int[2] padding=0, int[2] dilation=1,
                  bool ceil_mode=False) -> Tensor)SIG",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto kernel_size = params[1].toIntList().vec();
          auto stride = params[2].toIntList().vec();
          auto padding = params[3].toIntList().vec();
          auto dilation = params[4].toIntList().vec();
          auto ceil_mode = params[5].toBool();
          auto output_dtype = input_tensor->GetDataType();

          auto max_pool2d = handle->network->AddIMaxPool2DNode(input_tensor, false);
          MM_CHECK(max_pool2d->SetOutputType(0, output_dtype));

          MM_CHECK(max_pool2d->SetLayout(magicmind::Layout::NCHW, magicmind::Layout::NCHW));
          MM_CHECK(max_pool2d->SetKernel(kernel_size[0], kernel_size[1]));
          // stride is optional
          if (!stride.empty()) {
            MM_CHECK(max_pool2d->SetStride(stride[0], stride[1]));
          } else {
            // pytorch default stride value is kernel size
            MM_CHECK(max_pool2d->SetStride(kernel_size[0], kernel_size[1]));
          }
          MM_CHECK(max_pool2d->SetPad(padding[0], padding[0], padding[1], padding[1]));
          MM_CHECK(max_pool2d->SetDilation(dilation[0], dilation[1]));
          MM_CHECK(max_pool2d->SetCeilMode(ceil_mode));
          MM_CHECK(max_pool2d->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));

          auto output_tensor = max_pool2d->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op(R"SIG(aten::avg_pool2d(Tensor self, int[2] kernel_size,
                  int[2] stride=[], int[2] padding=0, bool ceil_mode=False,
                  bool count_include_pad=True, int? divisor_override=None) -> Tensor)SIG",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto kernel_size = params[1].toIntList().vec();
          auto stride = params[2].toIntList().vec();
          auto padding = params[3].toIntList().vec();
          auto ceil_mode = params[4].toBool();
          auto count_include_pad = params[5].toBool();
          auto output_dtype = input_tensor->GetDataType();

          auto avg_pool2d = handle->network->AddIAvgPool2DNode(input_tensor);
          MM_CHECK(avg_pool2d->SetOutputType(0, output_dtype));

          MM_CHECK(avg_pool2d->SetLayout(magicmind::Layout::NCHW, magicmind::Layout::NCHW));
          MM_CHECK(avg_pool2d->SetKernel(kernel_size[0], kernel_size[1]));
          // stride is optional
          if (!stride.empty()) {
            MM_CHECK(avg_pool2d->SetStride(stride[0], stride[1]));
          } else {
            // pytorch default stride value is kernel size
            MM_CHECK(avg_pool2d->SetStride(kernel_size[0], kernel_size[1]));
          }
          MM_CHECK(avg_pool2d->SetPad(padding[0], padding[0], padding[1], padding[1]));
          MM_CHECK(avg_pool2d->SetCeilMode(ceil_mode));
          MM_CHECK(avg_pool2d->SetCountIncludePad(count_include_pad));
          MM_CHECK(avg_pool2d->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));

          auto output_tensor = avg_pool2d->GetOutput(0);
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto out_size = params[1].toIntList().vec();
          auto output_dtype = input_tensor->GetDataType();

          auto adapt_avg_pool2d = handle->network->AddIAdaptiveAvgPool2DNode(input_tensor);
          MM_CHECK(adapt_avg_pool2d->SetOutputType(0, output_dtype));
          MM_CHECK(adapt_avg_pool2d->SetOutputSize(out_size[0], out_size[1]));
          MM_CHECK(adapt_avg_pool2d->SetLayout(
                      magicmind::Layout::NCHW, magicmind::Layout::NCHW));

          auto output_tensor = adapt_avg_pool2d->GetOutput(0);
          handle->bindingValueAndIvalue(node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        });

}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
