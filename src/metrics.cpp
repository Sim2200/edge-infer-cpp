#include "edgeinfer/metrics.hpp"

#include <algorithm>
#include <cstdio>

namespace edgeinfer {

Histogram::Histogram(std::vector<double> upper_bounds) : bounds_(std::move(upper_bounds)), counts_(bounds_.size() + 1) {}

void Histogram::observe(double value) {
  // Non-cumulative per-bucket counts; render() accumulates them. One atomic increment per sample.
  const auto it = std::lower_bound(bounds_.begin(), bounds_.end(), value);
  counts_[static_cast<std::size_t>(it - bounds_.begin())].fetch_add(1, std::memory_order_relaxed);
  total_.fetch_add(1, std::memory_order_relaxed);
  double cur = sum_.load(std::memory_order_relaxed);
  while (!sum_.compare_exchange_weak(cur, cur + value, std::memory_order_relaxed)) {
  }
}

void Histogram::render(const std::string& name, const std::string& help, std::string& out) const {
  char line[256];
  out += "# HELP " + name + " " + help + "\n# TYPE " + name + " histogram\n";
  std::uint64_t cumulative = 0;
  for (std::size_t i = 0; i < bounds_.size(); ++i) {
    cumulative += counts_[i].load(std::memory_order_relaxed);
    std::snprintf(line, sizeof line, "%s_bucket{le=\"%g\"} %llu\n", name.c_str(), bounds_[i],
                  static_cast<unsigned long long>(cumulative));
    out += line;
  }
  cumulative += counts_.back().load(std::memory_order_relaxed);
  std::snprintf(line, sizeof line, "%s_bucket{le=\"+Inf\"} %llu\n%s_sum %.6f\n%s_count %llu\n", name.c_str(),
                static_cast<unsigned long long>(cumulative), name.c_str(), sum_.load(std::memory_order_relaxed),
                name.c_str(), static_cast<unsigned long long>(total_.load(std::memory_order_relaxed)));
  out += line;
}

void LatencyWindow::add(double ms) {
  std::lock_guard lock(mu_);
  samples_[next_] = ms;
  next_ = (next_ + 1) % samples_.size();
  filled_ = std::min(filled_ + 1, samples_.size());
}

std::array<double, 3> LatencyWindow::quantiles() const {
  std::vector<double> copy;
  {
    std::lock_guard lock(mu_);
    copy.assign(samples_.begin(), samples_.begin() + static_cast<std::ptrdiff_t>(filled_));
  }
  if (copy.empty()) return {0, 0, 0};
  auto q = [&](double p) {
    const auto k = static_cast<std::size_t>(p * static_cast<double>(copy.size() - 1));
    std::nth_element(copy.begin(), copy.begin() + static_cast<std::ptrdiff_t>(k), copy.end());
    return copy[k];
  };
  return {q(0.50), q(0.95), q(0.99)};
}

std::string ServerMetrics::render() const {
  std::string out;
  out.reserve(4096);
  char line[256];
  auto counter = [&](const char* name, const char* help, unsigned long long v) {
    std::snprintf(line, sizeof line, "# HELP %s %s\n# TYPE %s counter\n%s %llu\n", name, help, name, name, v);
    out += line;
  };
  counter("edgeinfer_requests_total", "Requests accepted", requests_total.load());
  counter("edgeinfer_rejected_total", "Requests rejected because the queue was full (HTTP 503)", rejected_total.load());
  counter("edgeinfer_errors_total", "Requests that failed during inference", errors_total.load());
  counter("edgeinfer_batches_total", "Batched inference calls", batches_total.load());
  std::snprintf(line, sizeof line, "# HELP edgeinfer_queue_depth Requests waiting\n# TYPE edgeinfer_queue_depth gauge\nedgeinfer_queue_depth %lld\n",
                static_cast<long long>(queue_depth.load()));
  out += line;
  latency_seconds.render("edgeinfer_request_latency_seconds", "Enqueue to result, per request", out);
  queue_wait_seconds.render("edgeinfer_queue_wait_seconds", "Time a request waited before its batch started", out);
  batch_size.render("edgeinfer_batch_size", "Requests per inference call", out);
  const auto q = window.quantiles();
  std::snprintf(line, sizeof line,
                "# HELP edgeinfer_latency_ms Recent request latency quantiles (last 4096 requests)\n# TYPE edgeinfer_latency_ms gauge\n"
                "edgeinfer_latency_ms{quantile=\"0.5\"} %.3f\nedgeinfer_latency_ms{quantile=\"0.95\"} %.3f\nedgeinfer_latency_ms{quantile=\"0.99\"} %.3f\n",
                q[0], q[1], q[2]);
  out += line;
  return out;
}

}  // namespace edgeinfer
