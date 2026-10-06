#pragma once
// Dynamic micro-batching. Requests go into a BoundedQueue; `workers` threads each loop:
//
//   1. block until one request arrives (the batch's first request starts the clock),
//   2. keep taking requests until the batch has `max_batch` of them or `max_wait` has passed
//      since the first one arrived,
//   3. run ONE inference for the whole batch, split the logits, fulfil each request's promise.
//
// Batching trades latency for throughput: a lone request waits up to `max_wait` for company, but
// under load each inference call does more useful work per fixed overhead (graph launch, weight
// reads from memory). max_batch = 1 turns batching off and is the baseline in every benchmark.
//
// The worker is generic over the payload: the model-specific part (how to concatenate requests
// and how to call the model) is the `RunBatch` function given to the constructor.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "edgeinfer/bounded_queue.hpp"
#include "edgeinfer/metrics.hpp"

namespace edgeinfer {

struct Payload {
  std::vector<float> image;           // vision: 3*H*W floats, already preprocessed
  std::vector<std::int64_t> tokens;   // text: token ids (attention mask is implied: all ones)
};

struct Result {
  std::vector<float> logits;
  std::size_t batch_size = 0;   // how many requests shared the inference call
  double queue_wait_ms = 0;
  double infer_ms = 0;
};

struct Request {
  Payload payload;
  std::promise<Result> done;
  std::chrono::steady_clock::time_point enqueued = std::chrono::steady_clock::now();
};

struct BatcherConfig {
  std::size_t max_batch = 8;
  std::chrono::microseconds max_wait{2000};
  std::size_t workers = 1;
  std::size_t queue_capacity = 256;
};

class Batcher {
 public:
  // Runs one batch: payloads in, n * out_dim logits out.
  using RunBatch = std::function<std::vector<float>(const std::vector<Payload*>& batch)>;

  Batcher(BatcherConfig cfg, std::size_t out_dim, RunBatch run, ServerMetrics* metrics = nullptr);
  ~Batcher();
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  // Enqueue without blocking; an empty future means the queue was full (caller returns 503).
  std::optional<std::future<Result>> submit(Payload payload);
  void stop();
  const BatcherConfig& config() const { return cfg_; }

 private:
  void worker_loop();

  BatcherConfig cfg_;
  std::size_t out_dim_;
  RunBatch run_;
  ServerMetrics* metrics_;
  BoundedQueue<std::unique_ptr<Request>> queue_;
  std::vector<std::thread> workers_;
};

}  // namespace edgeinfer
