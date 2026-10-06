#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "edgeinfer/bounded_queue.hpp"

using namespace edgeinfer;

TEST(BoundedQueueTest, FifoOrder) {
  BoundedQueue<int> q(10);
  q.push(1);
  q.push(2);
  q.push(3);

  EXPECT_EQ(q.pop(), 1);
  EXPECT_EQ(q.pop(), 2);
  EXPECT_EQ(q.pop(), 3);
}

TEST(BoundedQueueTest, TryPush) {
  BoundedQueue<int> q(2);
  EXPECT_TRUE(q.try_push(1));
  EXPECT_TRUE(q.try_push(2));
  EXPECT_FALSE(q.try_push(3));  // full

  q.pop();
  EXPECT_TRUE(q.try_push(3));  // succeeds after pop
}

TEST(BoundedQueueTest, PopFor) {
  BoundedQueue<int> q(10);

  // Timeout on empty
  auto start = std::chrono::steady_clock::now();
  auto result = q.pop_for(std::chrono::milliseconds(100));
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start).count();

  EXPECT_FALSE(result);
  EXPECT_GE(elapsed, 100);
  EXPECT_LE(elapsed, 300);  // allow some tolerance
}

TEST(BoundedQueueTest, Close) {
  BoundedQueue<int> q(10);
  q.push(1);
  q.push(2);
  q.close();

  // push/try_push should fail
  EXPECT_FALSE(q.push(3));
  EXPECT_FALSE(q.try_push(4));

  // pop should drain existing
  EXPECT_EQ(q.pop(), 1);
  EXPECT_EQ(q.pop(), 2);
  EXPECT_FALSE(q.pop());  // nullopt after drained
}

TEST(BoundedQueueTest, DrainInto) {
  BoundedQueue<int> q(10);
  q.push(1);
  q.push(2);
  q.push(3);
  q.push(4);
  q.push(5);

  std::vector<int> out;
  std::size_t n = q.drain_into(out, 3);

  EXPECT_EQ(n, 3);
  EXPECT_EQ(out.size(), 3);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
  EXPECT_EQ(out[2], 3);

  // Remaining items
  EXPECT_EQ(q.pop(), 4);
  EXPECT_EQ(q.pop(), 5);
}

TEST(BoundedQueueTest, StressTest) {
  BoundedQueue<int> q(64);
  const int total_items = 50000;
  const int num_producers = 4;
  const int num_consumers = 4;
  const int per_producer = total_items / num_producers;

  std::atomic<int> received_count(0);
  std::atomic<long long> received_sum(0);

  // Producers
  std::vector<std::thread> threads;
  for (int p = 0; p < num_producers; ++p) {
    threads.emplace_back([&q, p, per_producer] {
      for (int i = 0; i < per_producer; ++i) {
        int value = p * per_producer + i;
        q.push(value);
      }
    });
  }

  // Consumers
  for (int c = 0; c < num_consumers; ++c) {
    threads.emplace_back([&q, &received_count, &received_sum] {
      while (auto val = q.pop()) {
        received_sum.fetch_add(*val, std::memory_order_relaxed);
        received_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Wait for producers
  for (int p = 0; p < num_producers; ++p) {
    threads[p].join();
  }
  q.close();

  // Wait for consumers
  for (int c = num_producers; c < num_producers + num_consumers; ++c) {
    threads[c].join();
  }

  EXPECT_EQ(received_count, total_items);
  const long long expected_sum = static_cast<long long>(total_items) * (total_items - 1) / 2;
  EXPECT_EQ(received_sum, expected_sum);
}
