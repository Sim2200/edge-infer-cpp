#include "edgeinfer/batcher.hpp"

#include <exception>

namespace edgeinfer {

namespace {
double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

Batcher::Batcher(BatcherConfig cfg, std::size_t out_dim, RunBatch run, ServerMetrics* metrics)
    : cfg_(cfg), out_dim_(out_dim), run_(std::move(run)), metrics_(metrics), queue_(cfg.queue_capacity) {
  if (cfg_.max_batch == 0 || cfg_.workers == 0) throw std::invalid_argument("max_batch and workers must be >= 1");
  workers_.reserve(cfg_.workers);
  for (std::size_t i = 0; i < cfg_.workers; ++i) workers_.emplace_back([this] { worker_loop(); });
}

Batcher::~Batcher() { stop(); }

void Batcher::stop() {
  queue_.close();
  for (auto& w : workers_) {
    if (w.joinable()) w.join();
  }
}

std::optional<std::future<Result>> Batcher::submit(Payload payload) {
  auto req = std::make_unique<Request>();
  req->payload = std::move(payload);
  auto fut = req->done.get_future();
  if (!queue_.try_push(std::move(req))) {
    if (metrics_) metrics_->rejected_total.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }
  if (metrics_) {
    metrics_->requests_total.fetch_add(1, std::memory_order_relaxed);
    metrics_->queue_depth.fetch_add(1, std::memory_order_relaxed);
  }
  return fut;
}

void Batcher::worker_loop() {
  std::vector<std::unique_ptr<Request>> batch;
  batch.reserve(cfg_.max_batch);
  std::vector<Payload*> payloads;
  payloads.reserve(cfg_.max_batch);
  for (;;) {
    batch.clear();
    auto first = queue_.pop();  // blocks; nullopt only after stop()
    if (!first) return;
    batch.push_back(std::move(*first));
    const auto deadline = batch.front()->enqueued + cfg_.max_wait;
    // Fill: take whatever is already queued in one lock, then wait for more until the deadline.
    while (batch.size() < cfg_.max_batch) {
      if (queue_.drain_into(batch, cfg_.max_batch - batch.size()) > 0) continue;
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) break;
      auto next = queue_.pop_for(deadline - now);
      if (!next) break;
      batch.push_back(std::move(*next));
    }

    const auto start = std::chrono::steady_clock::now();
    payloads.clear();
    for (auto& r : batch) payloads.push_back(&r->payload);
    if (metrics_) {
      metrics_->queue_depth.fetch_sub(static_cast<std::int64_t>(batch.size()), std::memory_order_relaxed);
      metrics_->batches_total.fetch_add(1, std::memory_order_relaxed);
      metrics_->batch_size.observe(static_cast<double>(batch.size()));
    }
    try {
      std::vector<float> logits = run_(payloads);
      const auto end = std::chrono::steady_clock::now();
      const double infer_ms = ms_between(start, end);
      for (std::size_t i = 0; i < batch.size(); ++i) {
        Result res;
        res.logits.assign(logits.begin() + static_cast<std::ptrdiff_t>(i * out_dim_),
                          logits.begin() + static_cast<std::ptrdiff_t>((i + 1) * out_dim_));
        res.batch_size = batch.size();
        res.queue_wait_ms = ms_between(batch[i]->enqueued, start);
        res.infer_ms = infer_ms;
        if (metrics_) {
          metrics_->queue_wait_seconds.observe(res.queue_wait_ms / 1000.0);
          const double total_ms = ms_between(batch[i]->enqueued, end);
          metrics_->latency_seconds.observe(total_ms / 1000.0);
          metrics_->window.add(total_ms);
        }
        batch[i]->done.set_value(std::move(res));
      }
    } catch (...) {
      if (metrics_) metrics_->errors_total.fetch_add(batch.size(), std::memory_order_relaxed);
      for (auto& r : batch) r->done.set_exception(std::current_exception());
    }
  }
}

}  // namespace edgeinfer
