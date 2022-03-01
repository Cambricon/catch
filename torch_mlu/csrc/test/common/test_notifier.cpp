#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

#include "aten/device/device.h"
#include "aten/device/notifier.h"
#include "aten/device/queue.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

TEST(NotifierTest, placeNotifier) {
  Notifier notifier;
  notifier.place();
  auto queue = getQueueFromPool();
  notifier.place(queue);
}

TEST(NotifierTest, syncNotifier) {
  Notifier notifier;
  notifier.place();
  notifier.synchronize();
}

TEST(NotifierTest, elapsed_time) {
  Notifier start;
  Notifier end;
  start.place();
  end.place();
  end.synchronize();
  float time = start.elapsed_time(end);
}

TEST(NotifierTest, queue_wait_notifier) {
  Notifier notifier;
  notifier.place();
  notifier.wait(getCurrentQueue());
  notifier.synchronize();
  notifier.place();
  notifier.wait(getQueueFromPool());
  notifier.synchronize();
}

TEST(NotifierTest, query_and_wait_notifier) {
  Notifier notifier;
  ASSERT_TRUE(notifier.query());
  notifier.synchronize();
  notifier.place();
  notifier.synchronize();
  ASSERT_TRUE(notifier.query());
}

TEST(NotifierTest, move_test) {
  Notifier no_1;
  Notifier no_2;
  no_2 = std::move(no_1);
  CHECK_EQ(no_1.device_index(), no_2.device_index());
  CHECK_EQ(no_1.isCreated(), no_2.isCreated());
}

TEST(NotifierTest, notifier_sync_test) {
  auto queue = getQueueFromPool();
  Notifier no;

  ASSERT_TRUE(no.query());

  no.placeOnce(queue);

  auto wait_queue0 = getQueueFromPool();
  auto wait_queue1 = getQueueFromPool();

  no.wait(wait_queue0);
  no.wait(wait_queue1);

  wait_queue0.synchronize();
  ASSERT_TRUE(no.query());
}

}
