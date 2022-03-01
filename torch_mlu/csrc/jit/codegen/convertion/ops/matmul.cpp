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

magicmind::ITensor* quantized_linear_forward(
    codegen::MagicmindHandle* handle, magicmind::ITensor* input,
    magicmind::ITensor* weight, magicmind::ITensor* bias,
    const std::vector<float>& scale_data, int qmode, int qsymmetry,
    magicmind::DataType output_dtype) {
  // TODO: Support asymmetry quantization.
  if (qsymmetry != 1) {
    AT_ERROR("Asymmetry quantization will be supported later");
    return nullptr;
  }

  auto qlinear_op = handle->network->AddIMatMulNode(input, weight, bias);
  /* The following usage of transA and transB is from MM`s  interface_network.h:
     * Consider the last two dimensions of inputa and inputb are [a1, a2] and [b1, b2](1-D after
     * promotion). The result shape of matrix multiplication is [trans_a ? a1 : a2, trans_b ? b2 : b1] .
    But I tried true+true, true+false, false+true and false+false cases, only false+false cases could pass.
    Other cases will report infer shape failure.  The reason need to be analysed. 
    TODO. */
  MM_CHECK(qlinear_op->SetTransA(false));
  MM_CHECK(qlinear_op->SetTransB(true));
  // linear op does not have following 2 params, so set to default.
  MM_CHECK(qlinear_op->SetScalarAB(1.0));
  MM_CHECK(qlinear_op->SetScalarAB(1.0));

  // set precision mode for quantization
  size_t bitwidth = 8;
  bool b_per_axis = (qmode > 2 && qmode < 5) ? true : false;
  if (qmode == 1 || qmode == 3) {
    MM_CHECK(qlinear_op->SetPrecision(0, magicmind::DataType::QINT8));  // set dtype for input
    MM_CHECK(qlinear_op->SetPrecision(1, magicmind::DataType::QINT8));  // set dtype for weight
    bitwidth = 8;
  } else if (qmode == 2 || qmode == 4) {
    MM_CHECK(qlinear_op->SetPrecision(0, magicmind::DataType::QINT16));  // set dtype for input
    MM_CHECK(qlinear_op->SetPrecision(1, magicmind::DataType::QINT16));  // set dtype for weight
    bitwidth = 16;
  }
  handle->quantized_bitwidth = (int)bitwidth;
  MM_CHECK(qlinear_op->SetPrecision(2, output_dtype));  // set precision mode for bias

  auto q_input = qlinear_op->GetInput(0);
  auto q_weight = qlinear_op->GetInput(1);

  // get dynamic ranges for input and weight
  std::vector<magicmind::Range> input_ranges;
  std::vector<magicmind::Range> weight_ranges;
  utils::getDynamicRange(scale_data, qmode, qsymmetry, bitwidth,
          input_ranges, weight_ranges);

  // set dynamic ranges for input quantization
  MM_CHECK(q_input->SetDynamicRange(input_ranges[0], false));
  // set dynamic ranges for weight quantization
  if (!b_per_axis) {
    MM_CHECK(q_weight->SetDynamicRange(weight_ranges[0], false));
  } else {
    // TODO: For now, Magicmind does not support per_channel quantization on linear op.
    CNLOG(ERROR) << "For now, Magicmind does not support per_channel quantization on linear op.";
    return nullptr;
  }
  MM_CHECK(qlinear_op->SetOutputType(0, output_dtype));
  auto output_tensor = qlinear_op->GetOutput(0);

  // Be careful for running on fp16 inference. The node output dtype is not converted yet.
  // Need insert cast(fp32->fp16) at the moment for next node dtype inference.
  // Then after the magicmind opt pass, the redundant cast op will be eliminated.
  if (output_tensor->GetDataType() != output_dtype) {
    auto cast = handle->network->AddICastNode(output_tensor, output_dtype);
    return cast->GetOutput(0);
  }
  return output_tensor;
}

static auto registry = Registerer()
    .op("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto self = codegen::getOrCreateITensor(handle, params[0]);
          auto mat1 = codegen::getOrCreateITensor(handle, params[1]);
          auto mat2 = codegen::getOrCreateITensor(handle, params[2]);
          auto beta = params[3].isInt() ? params[3].toInt() : params[3].toDouble();
          auto alpha = params[4].isInt() ? params[4].toInt() : params[4].toDouble();
          auto output_dtype = self->GetDataType();
          auto addmm_op = handle->network->AddIMatMulNode(mat1, mat2, self);
          MM_CHECK(addmm_op->SetOutputType(0, output_dtype));

          /* The following usage of transA and transB is from MM`s  interface_network.h:
              * Consider the last two dimensions of inputa and inputb are [a1, a2] and [b1, b2](1-D after
              * promotion). The result shape of matrix multiplication is [trans_a ? a1 : a2, trans_b ? b2 : b1] .
            But I tried true+true, true+false, false+true and false+false cases, only false+false cases could pass.
            Other cases will report infer shape failure.  The reason need to be analysed.
            TODO. */
          MM_CHECK(addmm_op->SetTransA(false));
          MM_CHECK(addmm_op->SetTransB(false));

          MM_CHECK(addmm_op->SetScalarAB(alpha));
          MM_CHECK(addmm_op->SetScalarC(beta));

          auto output_tensor = addmm_op->GetOutput(0);

          // Be careful for running on fp16 inference. The node output dtype is not converted yet.
          // Need insert cast(fp32->fp16) at the moment for next node dtype inference.
          // Then after the magicmind opt pass, the redundant cast op will be eliminated.
          if (output_tensor->GetDataType() != self->GetDataType()) {
            auto cast = handle->network->AddICastNode(output_tensor, self->GetDataType());
            output_tensor = cast->GetOutput(0);
          }

          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
           torch::jit::Stack& params) -> bool {
          auto self = codegen::getOrCreateITensor(handle, params[0]);
          auto mat1 = codegen::getOrCreateITensor(handle, params[1]);
          auto mat2 = codegen::getOrCreateITensor(handle, params[2]);
          auto beta = params[3].isInt() ? params[3].toInt() : params[3].toDouble();
          auto alpha = params[4].isInt() ? params[4].toInt() : params[4].toDouble();
          auto output_dtype = self->GetDataType();
          auto addmm_op = handle->network->AddIMatMulNode(mat1, mat2, self);
          MM_CHECK(addmm_op->SetOutputType(0, output_dtype));
          /* The following usage of transA and transB is from MM`s  interface_network.h:
              * Consider the last two dimensions of inputa and inputb are [a1, a2] and [b1, b2](1-D after
              * promotion). The result shape of matrix multiplication is [trans_a ? a1 : a2, trans_b ? b2 : b1] .
            But I tried true+true, true+false, false+true and false+false cases, only false+false cases could pass.
            Other cases will report infer shape failure.  The reason need to be analysed. 
            TODO. */
          MM_CHECK(addmm_op->SetTransA(false));
          MM_CHECK(addmm_op->SetTransB(false));

          MM_CHECK(addmm_op->SetScalarAB(alpha));
          MM_CHECK(addmm_op->SetScalarC(beta));
          auto output_tensor = addmm_op->GetOutput(0);

          // Be careful for running on fp16 inference. The node output dtype is not converted yet.
          // Need insert cast(fp32->fp16) at the moment for next node dtype inference.
          // Then after the magicmind opt pass, the redundant cast op will be eliminated.
          if (output_tensor->GetDataType() != self->GetDataType()) {
            auto cast = handle->network->AddICastNode(output_tensor, self->GetDataType());
            output_tensor = cast->GetOutput(0);
          }

          handle->bindingValueAndIvalue(
              node->outputs()[0], codegen::bindITensor(output_tensor));
          return true;
        })
    .op("torch_mlu::linear",
        [](codegen::MagicmindHandle* handle, const torch::jit::Node* node,
            torch::jit::Stack& params) -> bool {
          auto input_tensor = codegen::getOrCreateITensor(handle, params[0]);
          auto weight_tensor = codegen::getOrCreateITensor(handle, params[1]);
          auto bias_tensor = codegen::getOrCreateITensor(handle, params[2]);
          auto output_dtype = input_tensor->GetDataType();
          auto dim1 = input_tensor->GetDimension();
          auto dim2 = weight_tensor->GetDimension();
          auto dim3 = bias_tensor->GetDimension();
          // achieve quantized params and set to quantized linear API
          auto q_scale = params[3].toTensor().cpu().to(at::kFloat);
          std::vector<float> scale_data(static_cast<float *>(q_scale.data_ptr()),
                  static_cast<float *>(q_scale.data_ptr()) + q_scale.numel());
          // q_mode is int type which indicate:
          // 1: qint8;
          // 2: qint16;
          // 3: qint8_per_axis;
          // 4: qint16_per_axis.
          auto q_mode = *static_cast<int *>(params[4].toTensor().cpu().data_ptr());
          // TODO: support asymmetric quantization later.
          int q_symmetry = 1;

          // create Magicmind Op
          magicmind::ITensor* output_tensor = nullptr;
          output_tensor = quantized_linear_forward(handle, input_tensor, weight_tensor, bias_tensor,
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
