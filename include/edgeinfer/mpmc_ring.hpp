#pragma once
// A lock-free bounded MPMC ring buffer (Dmitry Vyukov's design), for comparison with BoundedQueue.
//
// Each slot carries a sequence number. A producer claims position `pos` by CAS on `tail_`, but may
// only write the slot when slot.seq == pos (the slot is free for this lap); it then publishes with
// seq = pos + 1. A consumer claims `pos` on `head_` and may read when slot.seq == pos + 1; it frees
// the slot for the next lap with seq = pos + capacity. No locks, no ABA problem (the sequence number
// encodes the lap), and producers and consumers only contend on their own index.
//
// It is non-blocking: try_push / try_pop fail instead of waiting. Callers that want to wait must
// spin or back off, which is the real trade-off measured in bench/: under oversubscription a
// spinning consumer burns the CPU the producer needs, where a condition variable would sleep.
//
// Memory ordering: acquire on reading seq pairs with release on writing it, so the value written to
// the slot happens-before the other side sees the new sequence number. The index CAS can be relaxed
// because the slot's seq, not the index, carries the synchronisation.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

namespace edgeinfer {

// 64 on x86-64 and most aarch64 parts. Hard-coded rather than
// std::hardware_destructive_interference_size, which GCC warns about as ABI-unstable.
inline constexpr std::size_t kCacheLine = 64;

template <typename T>
class MpmcRing {
 public:
  explicit MpmcRing(std::size_t capacity) : mask_(capacity - 1), slots_(new Slot[capacity]) {
    if (capacity < 2 || (capacity & (capacity - 1)) != 0) {
      throw std::invalid_argument("MpmcRing capacity must be a power of two >= 2");
    }
    for (std::size_t i = 0; i < capacity; ++i) slots_[i].seq.store(i, std::memory_order_relaxed);
  }

  MpmcRing(const MpmcRing&) = delete;
  MpmcRing& operator=(const MpmcRing&) = delete;

  bool try_push(T value) {
    std::size_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
      Slot& slot = slots_[pos & mask_];
      const std::size_t seq = slot.seq.load(std::memory_order_acquire);
      const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
      if (diff == 0) {
        if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          slot.value = std::move(value);
          slot.seq.store(pos + 1, std::memory_order_release);
          return true;
        }
        // CAS failure reloaded pos; retry with the new tail.
      } else if (diff < 0) {
        return false;  // the slot still holds last lap's item: full
      } else {
        pos = tail_.load(std::memory_order_relaxed);  // another producer moved on
      }
    }
  }

  std::optional<T> try_pop() {
    std::size_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
      Slot& slot = slots_[pos & mask_];
      const std::size_t seq = slot.seq.load(std::memory_order_acquire);
      const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
      if (diff == 0) {
        if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          T value = std::move(slot.value);
          slot.seq.store(pos + mask_ + 1, std::memory_order_release);
          return value;
        }
      } else if (diff < 0) {
        return std::nullopt;  // empty
      } else {
        pos = head_.load(std::memory_order_relaxed);
      }
    }
  }

  std::size_t capacity() const { return mask_ + 1; }

 private:
  struct alignas(kCacheLine) Slot {
    std::atomic<std::size_t> seq{0};
    T value{};
  };

  const std::size_t mask_;
  std::unique_ptr<Slot[]> slots_;
  // head and tail on separate cache lines: producers and consumers would otherwise false-share.
  alignas(kCacheLine) std::atomic<std::size_t> head_{0};
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
};

}  // namespace edgeinfer
