#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include "aten/core/caching_allocator.h"
#include "aten/device/device.h"
#include "aten/device/queue.h"
#include "aten/device/queue_guard.h"
#include "c10/util/Optional.h"

namespace torch_mlu {

TEST(QueueTest, getCurrentQueueTest) {
  {
    auto queue = getCurrentQueue();
    auto default_queue = getDefaultQueue();
    CHECK_EQ(queue, default_queue);
  }
  for (int i = 0; i < device_count(); ++i) {
    auto queue = getCurrentQueue(i);
    auto default_queue = getDefaultQueue(i);
    CHECK_EQ(queue, default_queue);
  }
}

TEST(QueueTest, getCnrtQueueTest) {
  {
    auto queue = getCurQueue();
    auto default_queue = getDefaultQueue();
    CHECK_EQ(queue, default_queue.queue());
    syncQueue(queue);
  }
  for (int i = 0; i < device_count(); ++i) {
    auto queue = getCurQueue(i);
    auto default_queue = getDefaultQueue(i);
    CHECK_EQ(queue, default_queue.queue());
  }
}

TEST(QueueTest, CopyAndMoveTest) {
  int32_t device = -1;
  cnrtQueue_t cnrt_queue;

  auto copyQueue = getQueueFromPool();
  {
    auto queue = getQueueFromPool();
    device = queue.device_index();
    cnrt_queue = queue.queue();

    copyQueue = queue;

    CHECK_EQ(copyQueue.device_index(), device);
    CHECK_EQ(copyQueue.queue(), cnrt_queue);
  }

  CHECK_EQ(copyQueue.device_index(), device);
  CHECK_EQ(copyQueue.queue(), cnrt_queue);

  auto moveQueue = getQueueFromPool();
  {
    auto queue = getQueueFromPool();
    device = queue.device_index();
    cnrt_queue = queue.queue();

    moveQueue = std::move(queue);

    CHECK_EQ(moveQueue.device_index(), device);
    CHECK_EQ(moveQueue.queue(), cnrt_queue);
  }
  CHECK_EQ(moveQueue.device_index(), device);
  CHECK_EQ(moveQueue.queue(), cnrt_queue);
}

TEST(QueueTest, GetAndSetTest) {
  auto myQueue = getQueueFromPool();

  // sets and gets
  setCurrentQueue(myQueue);
  auto curQueue = getCurrentQueue();

  CHECK_EQ(myQueue, curQueue);

  // Gets, sets, and gets the default stream
  auto defaultQueue = getDefaultQueue();
  setCurrentQueue(defaultQueue);
  curQueue = getCurrentQueue();

  CHECK_NE(defaultQueue, myQueue);
  CHECK_EQ(curQueue, defaultQueue);
}

void thread_fun(at::optional<Queue>& cur_thread_stream, int device) {
  for (int i = 0; i < 50; i++) {
    auto new_stream = getQueueFromPool();
    setCurrentQueue(new_stream);
    cur_thread_stream = {getCurrentQueue()};
    CHECK_EQ(*cur_thread_stream, new_stream);
  }
}

TEST(QueueTest, MultithreadGetAndSetTest) {
  at::optional<Queue> s0, s1;
  std::thread t0{thread_fun, std::ref(s0), 0};
  std::thread t1{thread_fun, std::ref(s1), 0};
  t0.join();
  t1.join();
  auto cur_queue = getCurrentQueue();
  auto default_queue = getDefaultQueue();
  EXPECT_EQ(cur_queue, default_queue);
  EXPECT_NE(cur_queue, *s0);
  EXPECT_NE(cur_queue, *s1);
}

TEST(QueueTest, QueuePoolTest) {
  std::vector<Queue> queues{};
  for (int i = 0; i < 200; ++i) {
    queues.emplace_back(getQueueFromPool());
  }

  std::unordered_set<cnrtQueue_t> queue_set{};
  bool hasDuplicates = false;
  for (auto i = decltype(queues.size()){0}; i < queues.size(); ++i) {
    auto mlu_queue = queues[i].queue();
    auto result_pair = queue_set.insert(mlu_queue);
    if (!result_pair.second) hasDuplicates = true;
  }
  EXPECT_TRUE(hasDuplicates);
}

TEST(QueueTest, MultiMLUTest) {
  if (device_count() < 2) return;

  auto s0 = getQueueFromPool(0);
  auto s1 = getQueueFromPool(1);
  setCurrentQueue(s0);
  setCurrentQueue(s1);

  CHECK_EQ(s0, getCurrentQueue());
  setDevice(1);
  CHECK_EQ(s1, getCurrentQueue());
}

TEST(QueueTest, QueueGuardTest) {
  auto original_queue = getCurrentQueue();
  auto queue = getQueueFromPool();
  torch_mlu::mlu::MLUQueueGuard guard(queue);
  CHECK_EQ(queue, getCurrentQueue());
  CHECK_NE(original_queue, getCurrentQueue());
  CHECK_EQ(guard.current_queue(), getCurrentQueue());
  CHECK_EQ(guard.original_queue(), original_queue);
  auto nqueue = getQueueFromPool();
  guard.reset_queue(nqueue);
  CHECK_EQ(nqueue, getCurrentQueue());
}

TEST(QueueTest, QueueQuery) {
  const size_t size = 100 * 1024 * 1024;
  auto queue = getCurrentQueue();
  ASSERT_TRUE(queue.query());
  auto allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(at::GetAllocator(at::kMLU));
  auto src_ptr = allocator->allocate(size, 0);
  auto dst_ptr = allocator->allocate(size, 0);
  cnrtMemcpyAsync(dst_ptr.get(), src_ptr.get(), size, queue.queue(),
                  CNRT_MEM_TRANS_DIR_DEV2DEV);
  ASSERT_FALSE(queue.query());
  queue.synchronize();
  ASSERT_TRUE(queue.query());
}

}  // namespace torch_mlu
