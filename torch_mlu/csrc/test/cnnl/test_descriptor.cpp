#include <cnnl.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "aten/cnnl/cnnlDescriptors.h"

namespace torch_mlu {

struct TestData {
  int data = 0;
};

cnnlStatus_t test_creator(TestData** data_t) {
  *data_t = new TestData;  // NOLINT (have to use 'new' for gtest)
  (*data_t)->data = 1;
  return CNNL_STATUS_SUCCESS;
}

cnnlStatus_t test_deleter(TestData* data) {
  delete data;  // NOLINT (have to use 'delete' for gtest)
  return CNNL_STATUS_SUCCESS;
}

TEST(CnnlDescriptorDeleter, CnnlDescriptordeleter) {
  std::unique_ptr<TestData, CnnlDescriptorDeleter<TestData,
      &test_deleter>> data_ptr(new TestData);
  CHECK_EQ(data_ptr->data, 0);
  data_ptr = nullptr;  // Call delete function
}

TEST(CnnlDescriptor, CnnlDescriptor) {
  CnnlDescriptor<TestData, &test_creator, &test_deleter> test_desc;

  TestData * mut_desc_ptr = test_desc.mut_desc();
  CHECK_EQ(mut_desc_ptr->data, 1);

  TestData * desc_ptr = test_desc.desc();
  CHECK_EQ(desc_ptr->data, 1);
  desc_ptr->data = 2;  // rewrite desc

  const CnnlDescriptor<TestData, &test_creator, &test_deleter> * const_desc = &test_desc;
  TestData * const_desc_ptr = const_desc->desc();
  CHECK_EQ(const_desc_ptr->data, 2);  // get rewrited data
}

TEST(CnnlCommondesc, CnnlGetcnnlstrides) {
  at::Tensor self = at::ones({2, 1, 4, 5}).to(at::Device(at::Device::Type::MLU));
  int dim_ = self.dim();
  std::vector<int> shape(dim_, 1), stride(dim_, 1);
  for (int i = 0; i < dim_; i++) {
      shape[i] = self.sizes()[i];
      stride[i] = self.strides()[i];
  }
  auto res = get_cnnl_strides(shape, stride);
  for (int i = 0; i < dim_; i++) {
      if (self.sizes()[i] != 1) {
          CHECK_EQ(res[i], self.strides()[i]);
      } else {
          CHECK_EQ(res[i], 1);
      }
  }
}

}  // namespace torch_mlu

