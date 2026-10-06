#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

#include "edgeinfer/mpmc_ring.hpp"

using namespace edgeinfer;

TEST(MpmcRingTest, InvalidCapacity) {
  EXPECT_THROW(MpmcRing<int>(3), std::invalid_argument);  // not power of 2
  EXPECT_THROW(MpmcRing<int>(1), std::invalid_argument);  // < 2
}

TEST(MpmcRingTest, SingleThreadPushFull) {
  MpmcRing<int> ring(4);

  // Fill completely
  EXPECT_TRUE(ring.try_push(1));
  EXPECT_TRUE(ring.try_push(2));
  EXPECT_TRUE(ring.try_push(3));
  EXPECT_TRUE(ring.try_push(4));

  // Should be full now
  EXPECT_FALSE(ring.try_push(5));
}

TEST(MpmcRingTest, SingleThreadFifo) {
  MpmcRing<int> ring(8);

  EXPECT_TRUE(ring.try_push(10));
  EXPECT_TRUE(ring.try_push(20));
  EXPECT_TRUE(ring.try_push(30));

  auto v1 = ring.try_pop();
  EXPECT_TRUE(v1);
  EXPECT_EQ(*v1, 10);

  auto v2 = ring.try_pop();
  EXPECT_TRUE(v2);
  EXPECT_EQ(*v2, 20);

  auto v3 = ring.try_pop();
  EXPECT_TRUE(v3);
  EXPECT_EQ(*v3, 30);

  auto v4 = ring.try_pop();
  EXPECT_FALSE(v4);  // empty
}

TEST(MpmcRingTest, StressTest) {
  MpmcRing<int> ring(1024);
  const int total_items = 200000;
  const int num_producers = 4;
  const int num_consumers = 4;
  const int per_producer = total_items / num_producers;

  std::atomic<int> received_count(0);
  std::atomic<long long> received_sum(0);

  std::vector<std::thread> threads;

  // Producers
  for (int p = 0; p < num_producers; ++p) {
    threads.emplace_back([&ring, p, per_producer] {
      for (int i = 0; i < per_producer; ++i) {
        int value = p * per_producer + i;
        while (!ring.try_push(value)) {
          std::this_thread::yield();
        }
      }
    });
  }

  // Consumers
  for (int c = 0; c < num_consumers; ++c) {
    threads.emplace_back([&ring, &received_count, &received_sum, per_producer] {
      int received_local = 0;
      while (received_local < per_producer) {
        if (auto val = ring.try_pop()) {
          received_sum.fetch_add(*val, std::memory_order_relaxed);
          received_count.fetch_add(1, std::memory_order_relaxed);
          received_local++;
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(received_count, total_items);
  const long long expected_sum = static_cast<long long>(total_items) * (total_items - 1) / 2;
  EXPECT_EQ(received_sum, expected_sum);
}

TEST(MpmcRingTest, MoveOnlyType) {
  MpmcRing<std::unique_ptr<int>> ring(4);

  auto p1 = std::make_unique<int>(42);
  EXPECT_TRUE(ring.try_push(std::move(p1)));
  EXPECT_FALSE(p1);  // moved

  auto p2 = ring.try_pop();
  EXPECT_TRUE(p2);
  EXPECT_EQ(**p2, 42);
}
