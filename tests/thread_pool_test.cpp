#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "edgeinfer/thread_pool.hpp"

using namespace edgeinfer;

TEST(ThreadPoolTest, SubmitAndWait) {
  ThreadPool pool(4);

  std::vector<std::future<int>> futures;
  for (int i = 0; i < 1000; ++i) {
    auto fut = pool.submit([i] { return i * 2; });
    futures.push_back(std::move(fut));
  }

  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(futures[i].get(), i * 2);
  }
}

TEST(ThreadPoolTest, ExceptionPropagation) {
  ThreadPool pool(2);

  auto fut = pool.submit([] { throw std::runtime_error("test error"); });

  EXPECT_THROW(fut.get(), std::runtime_error);
}

TEST(ThreadPoolTest, Shutdown) {
  ThreadPool pool(2);

  std::atomic<int> count(0);
  std::vector<std::future<void>> futures;
  for (int i = 0; i < 100; ++i) {
    auto fut = pool.submit([&count] {
      count.fetch_add(1, std::memory_order_relaxed);
    });
    futures.push_back(std::move(fut));
  }

  pool.shutdown();

  // Wait for all tasks
  for (auto& f : futures) {
    f.get();
  }

  EXPECT_EQ(count, 100);
}

TEST(ThreadPoolTest, ShutdownIdempotent) {
  ThreadPool pool(2);

  auto fut = pool.submit([] { return 42; });
  EXPECT_EQ(fut.get(), 42);

  pool.shutdown();
  pool.shutdown();  // second call should be safe
}

TEST(ThreadPoolTest, SubmitAfterShutdown) {
  ThreadPool pool(2);

  pool.shutdown();

  EXPECT_THROW(pool.submit([] { return 42; }), std::runtime_error);
}
