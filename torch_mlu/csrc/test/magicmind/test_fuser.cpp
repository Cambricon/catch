#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "jit/partition/passes/segment_graph.h"
#include "jit/partition/customer_partition_manager.h"
#include "jit/utils/utils.h"


namespace torch_mlu {

struct SegmentGraphTest : testing::Test {
  static void SetUpTestCase() {
    jit::partition::RegisterCustomOperators();
  }

  static bool is_supported_map(
      const torch::jit::Node *ptNode, const std::vector<std::string>& supported_map) {
    auto op_name = jit::utils::getOpFullNameFromNode(ptNode);
    auto it = std::find(supported_map.begin(), supported_map.end(), op_name);
    if (it != supported_map.end()) {
      return true;
    }
    return false;
  }
  static bool is_supported_1(const torch::jit::Node* ptNode) {
    std::vector<std::string> supported_map =
        {"aten::add_.Tensor", "aten::mul.Tensor", "aten::add.Tensor",
         "prim::Constant", "torch_mlu::MLUFusionGroup"};
    return is_supported_map(ptNode, supported_map);
  }
  static bool is_supported_debug(const torch::jit::Node *ptNode) {
    std::vector<std::string> supported_map =
        {"aten::relu_", "aten::mul.Tensor", "aten::abs", "aten::div.Scalar",
         "prim::Constant", "torch_mlu::MLUFusionGroup"};
    return is_supported_map(ptNode, supported_map);
  }
};

// add_ would not cause forming two fusiongroup
TEST_F(SegmentGraphTest, testFusionAliasing1) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Tensor = aten::mul(%0, %1)
      %2 : Tensor = aten::mul(%2.1, %1)
      %3 : Tensor = aten::add_(%2, %1, %12)
      %4 : Tensor = aten::mul(%2, %1)
      %5 : Tensor = aten::add(%2, %4, %12)
      return (%5))IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  jit::partition::Partitioner partitioner(is_supported_1);
  partitioner.fuseNodesForMM(g);
  torch::jit::EliminateDeadCode(g);

  torch::jit::testing::FileCheck()
      .check("torch_mlu::MLUFusionGroup_0")
      ->check("return")
      ->check_not("torch_mlu::MLUFusionGroup_1")
      ->run(*g);
}

//    relu
//   /   |
// mul1  mul2
//    |    |
//   add(not supported)
// if relu is unsupported, mul1,mul2 will belong to two separate fusiongroup
TEST_F(SegmentGraphTest, segment_graph0) {
  auto graph = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(
      R"IR(
  graph(%x, %y):
    %1 = aten::relu_(%x)
    %z1 = aten::mul(%1, %y)
    %z2 = aten::mul(%1, %y)
    %1 : int = prim::Constant[value=1]()
    %d = aten::add(%z1, %z2, %1)
    return (%d))IR",
      graph.get());

  graph->lint();

  jit::partition::Partitioner partitioner(is_supported_debug);
  partitioner.fuseNodesForMM(graph);
  torch::jit::EliminateDeadCode(graph);
  torch::jit::testing::FileCheck().check("graph")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("prim::Constant[value=1]()")
                                 ->check("aten::add")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("aten::relu_")
                                 ->check("aten::mul")
                                 ->check("aten::mul")
                                 ->check("return")
                                 ->run(*graph);
}

TEST_F(SegmentGraphTest, segment_graph1) {
  auto graph = std::make_shared<torch::jit::Graph>();
  // split into two segments
  torch::jit::parseIR(
       R"IR(
    graph(%x):
      %z1 = aten::abs(%x)
      %z2 = aten::hardsigmoid(%z1)
      %z3 = aten::abs(%z2)
      return (%z3))IR",
      graph.get());

  graph->lint();
  jit::partition::Partitioner partitioner(is_supported_debug);
  partitioner.fuseNodesForMM(graph);
  torch::jit::EliminateDeadCode(graph);
  torch::jit::testing::FileCheck().check("graph")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("aten::hardsigmoid")
                                 ->check("torch_mlu::MLUFusionGroup_1")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("aten::abs")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_1")
                                 ->check("aten::abs")
                                 ->check("return")
                                 ->run(*graph);
}

TEST_F(SegmentGraphTest, segment_graph2) {
  auto graph = std::make_shared<torch::jit::Graph>();
  // split into three segments
  torch::jit::parseIR(
       R"IR(
    graph(%x, %y, %z):
      %a = aten::abs(%x)
      %b = aten::hardsigmoid(%a)
      %1 : int = prim::Constant[value=1]()
      %c = aten::mul(%b, %y)
      %d = aten::sub(%c, %z, %1)
      %e = aten::div(%d, %1)
      return (%e))IR",
      graph.get());

  graph->lint();
  jit::partition::Partitioner partitioner(is_supported_debug);
  partitioner.fuseNodesForMM(graph);
  torch::jit::EliminateDeadCode(graph);
  torch::jit::testing::FileCheck().check("graph")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("prim::Constant[value=1]()")
                                 ->check("aten::hardsigmoid")
                                 ->check("torch_mlu::MLUFusionGroup_1")
                                 ->check("aten::sub")
                                 ->check("torch_mlu::MLUFusionGroup_2")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_0")
                                 ->check("aten::abs")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_1")
                                 ->check("aten::mul")
                                 ->check("return")
                                 ->check("torch_mlu::MLUFusionGroup_2")
                                 ->check("prim::Constant[value=1]()")
                                 ->check("aten::div")
                                 ->check("return")
                                 ->run(*graph);
}

}  // namespace torch_mlu

