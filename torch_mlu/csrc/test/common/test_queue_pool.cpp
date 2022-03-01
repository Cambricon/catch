#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

TEST(QueuePoolTest, getCurrentQueue) {
  const int cid = 0;
  setDevice(cid);
  for (int dev = 0; dev < device_count(); ++dev) {
    auto queue = getQueueFromPool(dev);
    CHECK_EQ(queue.device_index(), dev);
  }
  auto queue = getQueueFromPool(-1);
  CHECK_EQ(queue.device_index(), cid);
}
}  // namespace torch_mlu
