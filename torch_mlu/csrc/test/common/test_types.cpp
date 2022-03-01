#include <gtest/gtest.h>

// #include <c10/core/impl/InlineDeviceGuard.h>
#include "aten/util/types.h"

namespace torch_mlu {

TEST(TypeRelatedTest, TestInvalid) {
  try {
    getCnnlDataType(caffe2::TypeMeta());
  } catch(std::runtime_error& e) {
    std::cout << e.what() << std::endl;
  }

  ASSERT_EQ(toCnrtDtype(caffe2::TypeMeta()), CNRT_INVALID);

  cnrtDataType_t cnrt_type = CNRT_INT8;
  ASSERT_EQ(at::ScalarType::Char, cnrtType2ScalarType(cnrt_type));
  ASSERT_EQ(at::ScalarType::Undefined, cnrtType2ScalarType(CNRT_INVALID));
  ASSERT_EQ(CNRT_INVALID, CnnlType2CnrtType(CNNL_DTYPE_INVALID));
}

}  // namespace torch_mlu
