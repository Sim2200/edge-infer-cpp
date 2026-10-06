#pragma once
// A bounded multi-producer multi-consumer queue: one mutex, two condition variables.
//
// Why bounded: an inference server must push back when it is saturated. An unbounded queue turns
// overload into unbounded latency and memory; a bounded one lets the HTTP layer reject early
// (try_push fails -> 503), which is the behaviour a client can actually handle.
//
// Why two condition variables: producers wait for "not full", consumers wait for "not empty".
// With one CV a pop would wake a waiting pop (useless) as often as a waiting push.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace edgeinfer {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  // Blocks while full. Returns false only if the queue was closed.
  bool push(T value) {
    std::unique_lock lock(mu_);
    not_full_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
    if (closed_) return false;
    items_.push_back(std::move(value));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Never blocks. Returns false when full or closed (the caller decides: retry, drop, 503).
  bool try_push(T value) {
    {
      std::lock_guard lock(mu_);
      if (closed_ || items_.size() >= capacity_) return false;
      items_.push_back(std::move(value));
    }
    not_empty_.notify_one();
    return true;
  }

  // Blocks until an item arrives or the queue is closed and drained (then nullopt).
  std::optional<T> pop() {
    std::unique_lock lock(mu_);
    not_empty_.wait(lock, [&] { return closed_ || !items_.empty(); });
    return take(lock);
  }

  // Waits at most `timeout`; nullopt on timeout or on closed-and-drained.
  template <typename Rep, typename Period>
  std::optional<T> pop_for(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lock(mu_);
    if (!not_empty_.wait_for(lock, timeout, [&] { return closed_ || !items_.empty(); })) return std::nullopt;
    return take(lock);
  }

  // Moves up to `max_items` already-queued items into `out` without waiting. Used by the batcher
  // to fill a batch in one lock acquisition instead of one per request.
  std::size_t drain_into(std::vector<T>& out, std::size_t max_items) {
    std::size_t n = 0;
    {
      std::lock_guard lock(mu_);
      while (n < max_items && !items_.empty()) {
        out.push_back(std::move(items_.front()));
        items_.pop_front();
        ++n;
      }
    }
    if (n > 0) not_full_.notify_all();
    return n;
  }

  // After close(): pushes fail, pops drain what is left and then return nullopt.
  void close() {
    {
      std::lock_guard lock(mu_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard lock(mu_);
    return items_.size();
  }
  std::size_t capacity() const { return capacity_; }
  bool closed() const {
    std::lock_guard lock(mu_);
    return closed_;
  }

 private:
  std::optional<T> take(std::unique_lock<std::mutex>& lock) {
    if (items_.empty()) return std::nullopt;  // closed and drained
    T value = std::move(items_.front());
    items_.pop_front();
    lock.unlock();
    not_full_.notify_one();
    return value;
  }

  const std::size_t capacity_;
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> items_;
  bool closed_ = false;
};

}  // namespace edgeinfer
