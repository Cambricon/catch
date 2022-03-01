#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "utils/utils.h"
#include "aten/util/assert_tensor.h"

const char* assert_eq_str = "detect error, diff and prec are not equal";
const char* assert_le_str = "detect error, cpu and mlu results are bigger than threshold";
const char* assert_te_str = "detect error, two tensors sizes are not equal";

namespace torch_mlu {

TEST(AssertTensor, assertEqual) {
    float diff = 0.003;
    float prec = 0.003;
    assertEqual(diff, prec);
    diff = 1.;
    try {
      assertEqual(diff, prec);
      TORCH_MLU_CHECK(false, "assertEqual exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 41);
      if (strcmp(err.c_str(), assert_eq_str) != 0) {
        TORCH_MLU_CHECK(false, "assertEqual exception catch failed!");
      }
    }
}

TEST(AssertTensor, assertLessEqual) {
    float diff = 0.003;
    float prec = 0.003;
    assertLessEqual(diff, prec);
    diff = 1.;
    try {
      assertLessEqual(diff, prec);
      TORCH_MLU_CHECK(false, "assertLessEqual exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 59);
      if (strcmp(err.c_str(), assert_le_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual exception catch failed!");
      }
    }
}

TEST(AssertTensor, assertTensorsEqual) {
    auto a = at::ones((10)).to(at::Device(at::Device::Type::MLU));
    auto b = at::ones((5)).to(at::Device(at::Device::Type::MLU));

    // Detect wrong shapes
    try {
      assertTensorsEqual(a, b, 0.0, false, false, false);
      TORCH_MLU_CHECK(false, "assertLessEqual shape exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 45);
      if (strcmp(err.c_str(), assert_te_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual shape exception catch failed!");
      }
    }
    // Pass all cases
    float prec = 10.;
    b = at::zeros((10)).to(at::Device(at::Device::Type::MLU));
    assertTensorsEqual(a, b, prec, true, true, true);

    // Detect errors
    prec = 0.;
    try {
      assertTensorsEqual(a, b, prec, true, false, false);
      TORCH_MLU_CHECK(false, "assertLessEqual MSE mode exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 59);
      if (strcmp(err.c_str(), assert_le_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual MSE mode exception catch failed!");
      }
    }
    try {
      assertTensorsEqual(a, b, prec, false, true, false);
      TORCH_MLU_CHECK(false, "assertLessEqual RAE mode exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 59);
      if (strcmp(err.c_str(), assert_le_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual RAE mode exception catch failed!");
      }
    }
    try {
      assertTensorsEqual(a, b, prec, false, false, true);
      TORCH_MLU_CHECK(false, "assertLessEqual RMA mode exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 59);
      if (strcmp(err.c_str(), assert_le_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual RMA mode exception catch failed!");
      }
    }
    try {
      assertTensorsEqual(a, b, prec, true, false, false);
      TORCH_MLU_CHECK(false, "assertLessEqual ELSE mode exception catch failed!");
    } catch (std::exception& e) {
      std::string err = e.what();
      err = err.substr(0, 59);
      if (strcmp(err.c_str(), assert_le_str) != 0) {
        TORCH_MLU_CHECK(false, "assertLessEqual ELSE mode exception catch failed!");
      }
    }
}

}  // namespace torch_mlu
