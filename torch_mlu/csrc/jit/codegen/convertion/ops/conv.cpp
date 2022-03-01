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

bool is_depthwise_conv(int64_t groups, int64_t ci,
                       int64_t co, int64_t input_c) {
  return (groups != 1) && (co % groups == 0) &&
         (ci == 1) && (groups == input_c);
}

magicmind::ITensor* general_conv2d_forward(
    codegen::MagicmindHandle* handle, magicmind::ITensor* input,
    magicmind::ITensor* weight, magicmind::ITensor* bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  auto conv = handle->network->AddIConvNode(input, weight, bias);
  auto output_dtype = input->GetDataType();
  MM_CHECK(conv->SetLayout(magicmind::Layout::NCHW, magicmind::Layout::NCHW,
                           magicmind::Layout::NCHW));
  MM_CHECK(conv->SetStride(stride[0], stride[1]));
  MM_CHECK(conv->SetPad(padding[0], padding[0], padding[1], padding[1]));
  MM_CHECK(conv->SetDilation(dilation[0], dilation[1]));
  MM_CHECK(conv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
  MM_CHECK(conv->SetGroup(groups));
  MM_CHECK(conv->SetOutputType(0, output_dtype));

  auto conv_out = conv->GetOutput(0);
  // Be careful for running on fp16 inference. The node output dtype is not converted yet.
  // Need insert cast(fp32->fp16) at the moment for next node dtype inference.
  // Then after the magicmind opt pass, the redundant cast op will be eliminated.
  if (input->GetDataType() == conv_out->GetDataType()) return conv->GetOutput(0);
  auto cast = handle->network->AddICastNode(conv_out, output_dtype);
  return cast->GetOutput(0);
}

magicmind::ITensor* quantized_conv2d_forward(
    codegen::MagicmindHandle* handle, magicmind::ITensor* input,
    magicmind::ITensor* weight, magicmind::ITensor* bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, int64_t groups,
    const std::vector<float>& scale_data, int qmode, int qsymmetry,
    magicmind::DataType output_dtype) {
  // TODO: Support asymmetry quantization.
  if (qsymmetry != 1) {
    AT_ERROR("Asymmetry quantization will be supported later");
    return nullptr;
  }
  auto qconv = handle->network->AddIConvNode(input, weight, bias);
  MM_CHECK(qconv->SetLayout(magicmind::Layout::NCHW,
                           magicmind::Layout::NCHW,
                           magicmind::Layout::NCHW));
  MM_CHECK(qconv->SetStride(stride[0], stride[1]));
  MM_CHECK(qconv->SetPad(padding[0], padding[0], padding[1], padding[1]));
  MM_CHECK(qconv->SetDilation(dilation[0], dilation[1]));
  MM_CHECK(qconv->SetPaddingMode(magicmind::IPaddingMode::EXPLICIT));
  MM_CHECK(qconv->SetGroup(groups));
  size_t bitwidth = 8;
  bool b_per_axis = (qmode > 2 && qmode < 5) ? true : false;

  // set precision mode for quantization
  if (qmode == 1 || qmode == 3) {
    MM_CHECK(qconv->SetPrecision(0, magicmind::DataType::QINT8));  // set dtype for input
    MM_CHECK(qconv->SetPrecision(1, magicmind::DataType::QINT8));  // set dtype for weight
    bitwidth = 8;
  } else if (qmode == 2 || qmode == 4) {
    MM_CHECK(qconv->SetPrecision(0, magicmind::DataType::QINT16));  // set dtype for input
    MM_CHECK(qconv->SetPrecision(1, magicmind::DataType::QINT16));  // set dtype for weight
    bitwidth = 16;
  }
  handle->quantized_bitwidth = (int)bitwidth;
  MM_CHECK(qconv->SetPrecision(2, output_dtype));  //set dtype for bias

  auto q_input = qconv->GetInput(0);
  auto q_weight = qconv->GetInput(1);

  // get dynamic ranges for input and weight
  std::vector<magicmind::Range> input_ranges;
  std::vector<magicmind::Range> weight_ranges;
  utils::getDynamicRange(scale_data, qmode, qsymmetry, bitwidth,
          input_ranges, weight_ranges);

  // set dynamic ranges for input quantization
  MM_CHECK(q_input->SetDynamicRange(input_ranges[0], false));
  // set dynamic ranges for weight quantization
  if (!b_per_axis)
    MM_CHECK(q_weight->SetDynamicRange(weight_ranges[0], false));
  else
    MM_CHECK(q_weight->SetDynamicRangePerAxis(weight_ranges, false));

  MM_CHECK(qconv->SetOutputType(0, output_dtype));
  auto qconv_out = qconv->GetOutput(0);

  // Be careful for running on fp16 inference. The node output dtype is not converted yet.
  // Need insert cast(fp32->fp16) at the moment for next node dtype inference.
  // Then after the magicmind opt pass, the redundant cast op will be eliminated.
  if (output_dtype == qconv_out->GetDataType()) return qconv_out;
  auto cast = handle->network->AddICastNode(qconv_out, output_dtype);
  return cast->GetOutput(0);
}

magicmind::ITensor* depthwise_conv2d_forward(
    codegen::MagicmindHandle* handle, magicmind::ITensor* input,
    magicmind::ITensor* weight, magicmind::ITensor* bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  // TODO: support depthwise conv
  AT_ERROR("depthwise conv will be supported later.");
  return nullptr;
}

static auto registry = Registerer()
    .op(R"SIG(aten::_convolution(Tensor input, Tensor weight,
                  Tensor? bias, int[] stride, int[] padding,
                  int[] dilation, bool transposed,
                  int[] output_padding, int groups, bool benchmark,
                  bool deterministic, bool cudnn_enabled) -> (Tensor))SIG",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto weight_tensor = codegen::getOrCreateITensor(handle, params[1]);
          auto bias_tensor = params[2].isTensor() ? codegen::getOrCreateITensor(handle, params[2])
                             : nullptr;

          auto stride = params[3].toIntList().vec();
          auto padding = params[4].toIntList().vec();
          auto dilation = params[5].toIntList().vec();
          auto transposed = params[6].toBool();
          auto output_padding = params[7].toIntList().vec();
          auto groups = params[8].toInt();

          auto dim = weight_tensor->GetDimension().GetDimsNum();
          magicmind::ITensor* output_tensor = nullptr;
          if (dim == 4) {
            output_tensor = general_conv2d_forward(handle, input_tensor, weight_tensor,
                                                   bias_tensor, stride, padding, dilation,
                                                   transposed, output_padding, groups);
          } else {
            // TODO: support other conv3d mode laster
            AT_ERROR("Current conv mode is not supported.");
          }

          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("torch_mlu::conv2d",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto weight_tensor = codegen::getOrCreateITensor(handle, params[1]);
          auto bias_tensor = codegen::getOrCreateITensor(handle, params[2]);
          auto output_dtype = input_tensor->GetDataType();
          auto padding = params[3].toIntList().vec();
          auto stride = params[4].toIntList().vec();
          auto dilation = params[5].toIntList().vec();
          auto groups = params[6].toInt();
          // achieve quantized params and set to quantized convolution API.
          auto q_scale = params[7].toTensor().cpu().to(at::kFloat);
          // q_mode is int type which indicate:
          // 1: qint8;
          // 2: qint16;
          // 3: qint8_per_axis;
          // 4: qint16_per_axis.
          auto q_mode = *static_cast<int *>(params[8].toTensor().cpu().data_ptr());
          // q_symmetry is int type wihich indicate:
          // 1: symmetry quantization;
          // 0: asymmetry quantization.
          // TODO: For now, only support symmetry quantization.
          int q_symmetry = 1;
          std::vector<float> scale_data(static_cast<float *>(q_scale.data_ptr()),
                  static_cast<float *>(q_scale.data_ptr()) + q_scale.numel());

          // Create Magicmind Op
          magicmind::ITensor* output_tensor = nullptr;
          output_tensor = quantized_conv2d_forward(handle, input_tensor, weight_tensor,
                                                   bias_tensor, stride, padding, dilation, groups,
                                                   scale_data, q_mode, q_symmetry, output_dtype);
          if (output_tensor == nullptr) return false;
          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        });
}  // namespace convertion
}  // namespace codegen
}  // namespace jit
}  // namespace torch_mlu
