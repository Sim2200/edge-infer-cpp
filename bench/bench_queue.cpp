#include <benchmark/benchmark.h>
#include <thread>
#include <vector>

#include "edgeinfer/bounded_queue.hpp"
#include "edgeinfer/mpmc_ring.hpp"

using namespace edgeinfer;

static void BM_BoundedQueue(benchmark::State& state) {
  const int P = static_cast<int>(state.range(0));
  const int total_items = 1000000;
  const int per_thread = total_items / P;

  for (auto _ : state) {
    BoundedQueue<int> q(1024);

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    std::atomic<int> consumed(0);

    // Producers
    for (int p = 0; p < P; ++p) {
      producers.emplace_back([&q, p, per_thread] {
        for (int i = 0; i < per_thread; ++i) {
          q.push(p * per_thread + i);
        }
      });
    }

    // Consumers
    for (int c = 0; c < P; ++c) {
      consumers.emplace_back([&q, &consumed] {
        while (auto val = q.pop()) {
          consumed.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    // Wait for producers
    for (auto& t : producers) {
      t.join();
    }
    q.close();

    // Wait for consumers
    for (auto& t : consumers) {
      t.join();
    }
  }

  state.SetItemsProcessed(total_items * state.iterations());
  state.SetBytesProcessed(sizeof(int) * total_items * state.iterations());
}

static void BM_MpmcRing(benchmark::State& state) {
  const int P = static_cast<int>(state.range(0));
  const int total_items = 1000000;
  const int per_thread = total_items / P;

  for (auto _ : state) {
    MpmcRing<int> ring(1024);

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    std::atomic<int> consumed(0);

    // Producers
    for (int p = 0; p < P; ++p) {
      producers.emplace_back([&ring, p, per_thread] {
        for (int i = 0; i < per_thread; ++i) {
          int value = p * per_thread + i;
          while (!ring.try_push(value)) {
            std::this_thread::yield();
          }
        }
      });
    }

    // Consumers
    for (int c = 0; c < P; ++c) {
      consumers.emplace_back([&ring, &consumed] {
        while (true) {
          if (auto val = ring.try_pop()) {
            consumed.fetch_add(1, std::memory_order_relaxed);
          } else if (consumed.load(std::memory_order_relaxed) >= 1000000) {
            break;
          } else {
            std::this_thread::yield();
          }
        }
      });
    }

    // Wait for producers
    for (auto& t : producers) {
      t.join();
    }

    // Wait for consumers
    for (auto& t : consumers) {
      t.join();
    }
  }

  state.SetItemsProcessed(total_items * state.iterations());
  state.SetBytesProcessed(sizeof(int) * total_items * state.iterations());
}

BENCHMARK(BM_BoundedQueue)->Args({1})->Args({2})->Args({4})->UseRealTime()->Iterations(3);
BENCHMARK(BM_MpmcRing)->Args({1})->Args({2})->Args({4})->UseRealTime()->Iterations(3);
