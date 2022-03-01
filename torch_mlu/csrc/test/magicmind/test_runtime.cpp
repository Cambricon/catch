#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "jit/partition/passes/segment_graph.h"
#include "jit/partition/customer_partition_manager.h"
#include "jit/runtime/runner_cache.h"
#include "jit/runtime/graph_runner.h"
#include "jit/utils/utils.h"


namespace torch_mlu {

struct RuntimeTest : testing::Test {
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
};

// test for unique graph structure(input shape/type erased, debugname reordered)
// is stored in RunnerCache
TEST_F(RuntimeTest, testRegisterFusionCachesKernel) {
  // Constructs two functionally equivalent graphs
  const auto graph0_string = R"IR(
    graph(%0 : Double(4, 5, 6),
          %1 : Double(4, 5, 6)):
      %2 : int = prim::Constant[value=1]()
      %c0 : Double(4, 5, 6) = aten::add(%0, %1, %2)
      %d0 : Double(4, 5, 6) = aten::add(%c0, %0, %2)
      return (%d0))IR";
  auto g0 = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph0_string, g0.get());

  const auto graph1_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %2 : int = prim::Constant[value=1]()
      %c1 : Float(2, 3, 4) = aten::add(%0, %1, %2)
      %d1 : Float(2, 3, 4) = aten::add(%c1, %0, %2)
      return (%d1))IR";
  auto g1 = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph1_string, g1.get());

  auto getFusionGroup = [](const std::shared_ptr<torch::jit::Graph>& graph) {
    const auto& nodes = graph->nodes();
    auto maybe_fusion_group =
        std::find_if(nodes.begin(), nodes.end(), [](const torch::jit::Node* node) {
          return node->kind() == at::Symbol::fromQualString("torch_mlu::MLUFusionGroup");
        });
    TORCH_CHECK(
        maybe_fusion_group != nodes.end(),
        "testRegisterFusionCachesKernel: could not create FusionGroup");
    return *maybe_fusion_group;
  };

  // Creates two alpha-equivalent fusion groups
  jit::partition::Partitioner partitioner(is_supported_1);
  partitioner.fuseNodesForMM(g0);
  torch::jit::EliminateDeadCode(g0);

  partitioner.fuseNodesForMM(g1);
  torch::jit::EliminateDeadCode(g1);

  auto fg0 = getFusionGroup(g0);
  auto fg1 = getFusionGroup(g1);

  // Registers both with the fusion compiler.
  const auto key0 = torch_mlu::jit::runtime::RunnerCache::registerMLUFusion(fg0);
  const auto key1 = torch_mlu::jit::runtime::RunnerCache::registerMLUFusion(fg1);

  // Because the graphs are alpha-equivalent, they should return the same key
  // and therefore share a MMGraphRunner
  ASSERT_EQ(key0, key1);
  auto maybe_spec0 = torch_mlu::jit::runtime::RunnerCache::retrieve(key0);
  AT_ASSERT(maybe_spec0);
  auto maybe_spec1 = torch_mlu::jit::runtime::RunnerCache::retrieve(key1);
  AT_ASSERT(maybe_spec1);
  ASSERT_EQ(*maybe_spec0, *maybe_spec1);


  torch::Tensor t0 = torch::randn({3, 4, 5});
  torch::Tensor t1 = torch::randn({2, 3, 4});
  torch::Tensor t2 = torch::randn({2, 3, 4});
  auto t0_mlu = t0.to("mlu");
  auto t1_mlu = t1.to("mlu");
  // test for device 1
  auto t2_mlu = t2.to(torch::Device(torch::kMLU, 1));
  auto t2_mlu_ = t2.to(torch::Device(torch::kMLU));
  torch::jit::Stack input0{t0, t0};
  torch::jit::Stack input1{t1, t1};
  torch::jit::Stack input2{t2, t2};

  torch::jit::Stack input0_mlu{t0_mlu, t0_mlu};
  torch::jit::Stack input1_mlu{t1_mlu, t1_mlu};
  torch::jit::Stack input2_mlu{t2_mlu, t2_mlu};
  torch::jit::Stack input2_mlu_{t2_mlu_, t2_mlu_};

  (*maybe_spec0)->runMLUFusion(input0_mlu);
  (*maybe_spec0)->runMLUFusion(input1_mlu);
  (*maybe_spec0)->runMLUFusion(input2_mlu);
  (*maybe_spec0)->runMLUFusion(input2_mlu_);

  (*maybe_spec0)->runFallback(input0);
  (*maybe_spec0)->runFallback(input1);
  (*maybe_spec0)->runFallback(input2);
  for (int i = 0; i < input0.size(); ++i) {
    ASSERT_TRUE(i < input1.size() && i < input1_mlu.size() && i < input0_mlu.size());
    TORCH_CHECK(input0[i].toTensor().equal(input0_mlu[i].toTensor().to("cpu")));
    TORCH_CHECK(input1[i].toTensor().equal(input1_mlu[i].toTensor().to("cpu")));
    TORCH_CHECK(input2[i].toTensor().equal(input2_mlu[i].toTensor().to("cpu")));
    TORCH_CHECK(input2[i].toTensor().equal(input2_mlu_[i].toTensor().to("cpu")));
  }
}

}  // namespace torch_mlu

