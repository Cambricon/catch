#include <torch/torch.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <interface_builder.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "jit/codegen/convertion/convert.h"
#include "jit/codegen/evalution/eval.h"
#include "jit/codegen/tensor.h"
#include "jit/codegen/io.h"
#include "jit/codegen/codegen.h"

namespace torch_mlu {

TEST(MagicMindCodegenModuleTest, NodeConvertTest) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Tensor = aten::mul(%0, %1)
      %2 : Tensor = aten::mul(%2.1, %1)
      %3 : Tensor = aten::add_(%2, %1, %12)
      %4 : Tensor = aten::relu(%3)
      return (%4))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, g.get());

  auto nodes = g->nodes();
  for (auto node : nodes) {
    if (node->kind().toUnqualString() == "relu") {
      ASSERT_EQ(jit::codegen::convertion::isConvertNode(node), true);
      ASSERT_EQ(jit::codegen::convertion::findOrGetNodeConvert(node) != nullptr, true);
    } else if (node->kind().toUnqualString() == "Constant") {
      ASSERT_EQ(jit::codegen::convertion::isConvertNode(node), false);
      ASSERT_EQ(jit::codegen::convertion::findOrGetNodeConvert(node) == nullptr, true);
    }
  }
}

TEST(MagicMindCodegenModuleTest, NodeEvalutionTest) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Tensor = aten::mul(%0, %1)
      %2 : Tensor = aten::mul(%2.1, %1)
      %3 : Tensor = aten::add_(%2, %1, %12)
      %4 : Tensor = aten::relu(%3)
      return (%4))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, g.get());

  auto nodes = g->nodes();
  for (auto node : nodes) {
    if (node->kind() == c10::Symbol::fromQualString("prim::Constant")) {
      ASSERT_EQ(jit::codegen::evalution::isEvalNode(node), true);
      ASSERT_EQ(jit::codegen::evalution::findOrGetNodeEval(node) != nullptr, true);
    } else {
      ASSERT_EQ(jit::codegen::evalution::isEvalNode(node), false);
      ASSERT_EQ(jit::codegen::evalution::findOrGetNodeEval(node) == nullptr, true);
    }
  }
}

TEST(MagicMindCodegenModuleTest, CreateConstITensorTest) {
  jit::codegen::MagicmindHandle handle;

  c10::ScalarType scalar_types_arrays[7] = {
    at::kFloat, at::kHalf, at::kInt, at::kChar,
    at::kBool, at::kLong, at::kDouble
  };

  for (auto scalar_type : scalar_types_arrays) {
    torch::Tensor input = torch::rand({2, 3}).to(scalar_type);

    auto const_itensor = jit::codegen::createConstITensor(&handle, input);
    switch (scalar_type) {
    case at::kFloat:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::FLOAT32);
      break;
    case at::kHalf:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::FLOAT16);
      break;
    case at::kInt:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::INT32);
      break;
    case at::kChar:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::INT8);
      break;
    case at::kBool:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::BOOL);
      break;
    case at::kLong:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::INT32);
      break;
    case at::kDouble:
      ASSERT_EQ(const_itensor->GetDataType(), magicmind::DataType::FLOAT32);
      break;
    default:
      break;
    }
  }
}

TEST(MagicMindCodegenModuleTest, GetorCreateITensorTest) {
  jit::codegen::MagicmindHandle handle;

  magicmind::DataType op_datatype = magicmind::DataType::FLOAT32;
  magicmind::Dims op_dims({64, 64, 64, 64});
  magicmind::ITensor *input_0 = handle.network->AddInput(op_datatype, op_dims);

  auto ivalue = jit::codegen::bindITensor(input_0);
  auto new_itensor = jit::codegen::getOrCreateITensor(&handle, ivalue);

  ASSERT_EQ(input_0, new_itensor);
}

TEST(MagicMindCodegenModuleTest, NetworkIOTest) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : Tensor = aten::relu(%0)
      return (%1))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, g.get());
  const torch::jit::Block* block = g->block();

  jit::codegen::MagicmindHandle handle;

  torch::jit::IValue input_1 = torch::rand({2, 3}).to(at::kFloat);
  torch::jit::Stack stack;
  stack.emplace_back(input_1);

  ASSERT_EQ(jit::codegen::addInputs(&handle, block->inputs(), stack), true);

  for (auto node : block->nodes()) {
    jit::codegen::convertion::convertNode(&handle, node);
  }
  ASSERT_EQ(jit::codegen::markOutputs(&handle, block->outputs()), true);
}

}  // namespace torch_mlu
