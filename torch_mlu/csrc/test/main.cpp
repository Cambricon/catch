#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include "aten/generated/aten_mlu_type_default.h"
#include "utils/utils.h"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  torch_mlu::RegisterAtenOperators();
  init();
  return RUN_ALL_TESTS();
}  //  main
