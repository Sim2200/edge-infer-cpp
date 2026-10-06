#pragma once
// Prometheus metrics without a client library: counters, gauges and fixed-bucket histograms with
// atomic updates, plus a sliding window of recent latencies for exact p50/p95/p99 gauges.
//
// Histograms answer "how is latency distributed" in Prometheus (and aggregate across replicas);
// the window answers "what is p95 right now" for the benchmark harness and a dashboard without a
// PromQL histogram_quantile. Both are exported.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace edgeinfer {

class Histogram {
 public:
  explicit Histogram(std::vector<double> upper_bounds);
  void observe(double value);
  // Appends "<name>_bucket{le=...}", "_sum" and "_count" lines in Prometheus text format.
  void render(const std::string& name, const std::string& help, std::string& out) const;

 private:
  std::vector<double> bounds_;
  std::vector<std::atomic<std::uint64_t>> counts_;  // one per bound, plus +Inf
  std::atomic<std::uint64_t> total_{0};
  std::atomic<double> sum_{0.0};
};

class LatencyWindow {
 public:
  explicit LatencyWindow(std::size_t capacity = 4096) : samples_(capacity) {}
  void add(double ms);
  // {p50, p95, p99} over the most recent `capacity` samples (zeros if empty).
  std::array<double, 3> quantiles() const;

 private:
  mutable std::mutex mu_;
  std::vector<double> samples_;
  std::size_t next_ = 0;
  std::size_t filled_ = 0;
};

struct ServerMetrics {
  std::atomic<std::uint64_t> requests_total{0};
  std::atomic<std::uint64_t> rejected_total{0};   // queue full -> HTTP 503
  std::atomic<std::uint64_t> errors_total{0};
  std::atomic<std::uint64_t> batches_total{0};
  std::atomic<std::int64_t> queue_depth{0};
  Histogram latency_seconds{{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0}};
  Histogram queue_wait_seconds{{0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0}};
  Histogram batch_size{{1, 2, 4, 8, 16, 32, 64}};
  LatencyWindow window;

  std::string render() const;  // the whole /metrics page
};

}  // namespace edgeinfer
