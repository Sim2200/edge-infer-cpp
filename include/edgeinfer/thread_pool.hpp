#pragma once
// A fixed-size thread pool on top of BoundedQueue. submit() returns a std::future for the result.
//
// The pool size is a deliberate, measured choice in this project, not "hardware_concurrency()":
// each inference already uses ONNX Runtime's own intra-op threads, so pool_threads * ort_threads
// must not exceed the cores the container is allowed, or the scheduler time-slices them and p95
// explodes (bench_cli --sweep-threads measures exactly that).

#include <cstddef>
#include <functional>
#include <future>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "edgeinfer/bounded_queue.hpp"

namespace edgeinfer {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t threads, std::size_t queue_capacity = 1024) : tasks_(queue_capacity) {
    workers_.reserve(threads);
    for (std::size_t i = 0; i < threads; ++i) {
      workers_.emplace_back([this] {
        while (auto task = tasks_.pop()) (*task)();
      });
    }
  }

  ~ThreadPool() { shutdown(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Runs `fn` on a worker. Blocks if the queue is full (back-pressure on the submitter).
  template <typename F>
  auto submit(F&& fn) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    // packaged_task is move-only; std::function needs copyable, so hold it in a shared_ptr.
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(fn));
    std::future<R> result = task->get_future();
    if (!tasks_.push([task] { (*task)(); })) {
      throw std::runtime_error("ThreadPool::submit after shutdown");
    }
    return result;
  }

  // Finishes queued tasks, then joins. Idempotent.
  void shutdown() {
    tasks_.close();
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

  std::size_t size() const { return workers_.size(); }

 private:
  BoundedQueue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
};

}  // namespace edgeinfer
