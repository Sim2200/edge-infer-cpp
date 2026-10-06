#include <gtest/gtest.h>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "edgeinfer/batcher.hpp"

using namespace edgeinfer;

// Fake RunBatch that returns the first byte of each image as logits
std::vector<float> fake_run_batch(const std::vector<Payload*>& batch) {
  std::vector<float> result;
  for (auto* p : batch) {
    if (!p->image.empty()) {
      result.push_back(p->image[0]);
    } else {
      result.push_back(0.0f);
    }
  }
  return result;
}

TEST(BatcherTest, SingleRequest) {
  BatcherConfig cfg;
  cfg.max_batch = 8;
  cfg.max_wait = std::chrono::milliseconds(2);
  cfg.workers = 1;

  Batcher batcher(cfg, 1, fake_run_batch);

  Payload p;
  p.image = {42.0f};
  auto fut = batcher.submit(p);

  EXPECT_TRUE(fut);
  auto result = fut->get();
  EXPECT_EQ(result.batch_size, 1);
  EXPECT_EQ(result.logits.size(), 1);
  EXPECT_FLOAT_EQ(result.logits[0], 42.0f);

  batcher.stop();
}

TEST(BatcherTest, MultipleBatches) {
  BatcherConfig cfg;
  cfg.max_batch = 8;
  cfg.max_wait = std::chrono::milliseconds(50);
  cfg.workers = 1;
  cfg.queue_capacity = 256;

  Batcher batcher(cfg, 1, fake_run_batch);

  std::vector<std::future<Result>> futures;
  for (int i = 0; i < 16; ++i) {
    Payload p;
    p.image = {static_cast<float>(i)};
    auto fut = batcher.submit(p);
    EXPECT_TRUE(fut);
    futures.push_back(std::move(*fut));
  }

  int batches_size_gt_1 = 0;
  for (int i = 0; i < 16; ++i) {
    auto result = futures[i].get();
    EXPECT_LE(result.batch_size, 8);
    EXPECT_GE(result.batch_size, 1);
    EXPECT_EQ(result.logits[0], static_cast<float>(i));
    if (result.batch_size > 1) batches_size_gt_1++;
  }

  EXPECT_GT(batches_size_gt_1, 0);  // At least one batch with size > 1

  batcher.stop();
}

TEST(BatcherTest, MaxBatch1) {
  BatcherConfig cfg;
  cfg.max_batch = 1;
  cfg.max_wait = std::chrono::milliseconds(10);
  cfg.workers = 1;

  Batcher batcher(cfg, 1, fake_run_batch);

  std::vector<std::future<Result>> futures;
  for (int i = 0; i < 10; ++i) {
    Payload p;
    p.image = {static_cast<float>(i)};
    auto fut = batcher.submit(p);
    EXPECT_TRUE(fut);
    futures.push_back(std::move(*fut));
  }

  for (int i = 0; i < 10; ++i) {
    auto result = futures[i].get();
    EXPECT_EQ(result.batch_size, 1);  // Always batch size 1
  }

  batcher.stop();
}

TEST(BatcherTest, QueueFull) {
  BatcherConfig cfg;
  cfg.max_batch = 8;
  cfg.max_wait = std::chrono::milliseconds(100);
  cfg.workers = 1;
  cfg.queue_capacity = 2;

  ServerMetrics metrics;

  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();  // every batch waits on the same signal
  auto blocking_run = [released](const std::vector<Payload*>& batch) -> std::vector<float> {
    released.wait();
    return std::vector<float>(batch.size(), 0.0f);  // one logit per request (out_dim 1)
  };

  Batcher batcher(cfg, 1, blocking_run, &metrics);

  // Submit one request to start processing (blocks RunBatch)
  Payload p1;
  p1.image = {1.0f};
  auto fut1 = batcher.submit(p1);
  EXPECT_TRUE(fut1);

  std::thread t([&] {
    // Wait for the first request to start processing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Now try to overflow the queue
    for (int i = 0; i < 10; ++i) {
      Payload p;
      p.image = {static_cast<float>(i)};
      auto fut = batcher.submit(p);
      if (!fut) {
        // Queue was full, rejected
        break;
      }
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  release.set_value();

  t.join();
  fut1->get();

  EXPECT_GT(metrics.rejected_total.load(), 0);

  batcher.stop();
}

TEST(BatcherTest, ErrorHandling) {
  BatcherConfig cfg;
  cfg.max_batch = 8;
  cfg.max_wait = std::chrono::milliseconds(10);
  cfg.workers = 1;

  ServerMetrics metrics;

  auto throwing_run = [](const std::vector<Payload*>&) -> std::vector<float> {
    throw std::runtime_error("test error");
  };

  Batcher batcher(cfg, 1, throwing_run, &metrics);

  Payload p;
  p.image = {42.0f};
  auto fut = batcher.submit(p);
  EXPECT_TRUE(fut);

  EXPECT_THROW(fut->get(), std::runtime_error);
  EXPECT_EQ(metrics.errors_total.load(), 1);

  batcher.stop();
}

TEST(BatcherTest, MetricsRender) {
  BatcherConfig cfg;
  cfg.max_batch = 8;
  cfg.max_wait = std::chrono::milliseconds(10);
  cfg.workers = 1;

  ServerMetrics metrics;
  Batcher batcher(cfg, 1, fake_run_batch, &metrics);

  Payload p;
  p.image = {42.0f};
  auto fut = batcher.submit(p);
  EXPECT_TRUE(fut);
  fut->get();

  std::string rendered = metrics.render();
  EXPECT_NE(rendered.find("edgeinfer_requests_total"), std::string::npos);
  EXPECT_NE(rendered.find("edgeinfer_batch_size_bucket"), std::string::npos);

  batcher.stop();
}

TEST(BatcherTest, WrongOutputSizeFailsTheBatch) {
  BatcherConfig cfg;
  cfg.max_batch = 4;
  cfg.max_wait = std::chrono::milliseconds(20);
  ServerMetrics metrics;
  // A model that returns one value no matter how many requests are in the batch.
  Batcher batcher(cfg, 1, [](const std::vector<Payload*>&) { return std::vector<float>{0.0f}; }, &metrics);
  std::vector<std::future<Result>> futs;
  for (int i = 0; i < 4; ++i) {
    Payload p;
    p.image = {1.0f};
    auto f = batcher.submit(std::move(p));
    ASSERT_TRUE(f);
    futs.push_back(std::move(*f));
  }
  int failed = 0;
  for (auto& f : futs) {
    try {
      const Result r = f.get();
      EXPECT_EQ(r.batch_size, 1u);  // a lone request still gets exactly the one value it needs
    } catch (const std::runtime_error&) {
      ++failed;
    }
  }
  EXPECT_GT(failed, 0);
  EXPECT_EQ(metrics.errors_total.load(), static_cast<std::uint64_t>(failed));
}

