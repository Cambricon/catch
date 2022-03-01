#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "aten/core/caching_allocator.h"
#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "aten/device/queue_guard.h"
#include "c10/util/Optional.h"

#include "utils/utils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

TEST(TensorUtilTest, getTensorUtilTest) {
  at::Tensor self = at::ones({1}).to(at::Device(at::Device::Type::MLU));
  auto self_impl = getMluTensorImpl(self);
  CHECK_EQ(self_impl->device_type(), c10::DeviceType::MLU);
}

TEST(TensorUtilTest, getTensorDeviceTest) {
  at::Tensor self = at::ones({1}).to(at::Device(at::Device::Type::MLU, 1));
  auto device_index = getTensorDevice({self});
  CHECK_EQ(device_index, 1);
}

TEST(TensorUtilTest, copy_to_cpu_cnnlTest) {
  at::Tensor self = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto self_cpu = at::empty_like(self, self.options().device(at::kCPU)).zero_();
  copy_to_cpu_cnnl(self_cpu, self);
  CHECK_EQ(self_cpu.device(), c10::DeviceType::CPU);
  assertTensorsEqual(self_cpu, self.cpu(), 0.0, true, false, false);
}

TEST(TensorUtilTest, ismluTest) {
  at::Tensor self = at::ones({2, 4}).to(at::Device(at::Device::Type::MLU));
  auto self_impl = getMluTensorImpl(self);
  bool self_ismlu = is_mlu(self);
  bool self_impl_ismlu = is_mlu(self_impl);
  CHECK_EQ(self_ismlu, true);
  CHECK_EQ(self_impl_ismlu, true);
}

TEST(TensorUtilTest, ischannellastTest) {
  at::Tensor self = at::ones({2, 4, 3, 5}).to(at::Device(at::Device::Type::MLU));
  at::Tensor self_cl = self.to(at::MemoryFormat::ChannelsLast);
  bool self_iscl = is_channels_last(self);
  bool self_cl_iscl = is_channels_last(self_cl);
  CHECK_EQ(self_iscl, false);
  CHECK_EQ(self_cl_iscl, true);
}

}  // namespace torch_mlu
