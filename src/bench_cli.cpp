// edge_bench: direct (no HTTP) measurements of one model, written as JSON.
//
//   edge_bench latency --model M --batches 1,8 --intra 4 --iters 50 --out results/x.json
//     Load time, single-call latency per batch size (p50/p95, per-item), peak RSS of this process.
//   edge_bench threads --model M --callers 1,2,4,8 --intra 1,2,4,8 --seconds 5 --out results/y.json
//     C concurrent callers, each running batch-1 inference on one shared session with T intra-op
//     threads: throughput and p95 for every (C, T). C * T above the core count is oversubscription.
//     --spin also measures allow_spinning=1 at the same settings.
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "edgeinfer/model.hpp"

using namespace edgeinfer;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<int> int_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
  return out;
}

double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
}

struct Inputs {
  std::vector<float> image;           // one preprocessed image
  std::vector<std::int64_t> ids, mask;  // one sequence
  std::size_t seq = 128;
};

Inputs make_inputs(const Model& m, std::size_t batch) {
  Inputs in;
  std::mt19937 rng(7);
  if (m.kind() == ModelKind::Vision) {
    std::normal_distribution<float> d(0.f, 1.f);
    in.image.resize(batch * 3 * static_cast<std::size_t>(m.input_size()) * static_cast<std::size_t>(m.input_size()));
    for (auto& v : in.image) v = d(rng);
  } else {
    std::uniform_int_distribution<std::int64_t> tok(1000, 29000);
    in.ids.resize(batch * in.seq);
    in.mask.assign(batch * in.seq, 1);
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t t = 0; t < in.seq; ++t) in.ids[b * in.seq + t] = tok(rng);
      in.ids[b * in.seq] = 101;                // [CLS]
      in.ids[b * in.seq + in.seq - 1] = 102;   // [SEP]
    }
  }
  return in;
}

void run_once(const Model& m, const Inputs& in, std::size_t batch) {
  if (m.kind() == ModelKind::Vision) (void)m.run_vision(in.image.data(), batch);
  else (void)m.run_text(in.ids.data(), in.mask.data(), batch, in.seq);
}

std::string arg(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 2; i + 1 < argc; ++i) if (argv[i] == key) return argv[i + 1];
  return def;
}
bool flag(int argc, char** argv, const std::string& key) {
  for (int i = 2; i < argc; ++i) if (argv[i] == key) return true;
  return false;
}

json latency(int argc, char** argv) {
  ModelConfig mc;
  mc.path = arg(argc, argv, "--model", "");
  mc.intra_op_threads = std::stoi(arg(argc, argv, "--intra", "1"));
  const int iters = std::stoi(arg(argc, argv, "--iters", "50"));
  const double rss_before = peak_rss_mib();
  const auto t0 = Clock::now();
  Model m(mc);
  const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  json rows = json::array();
  for (int b : int_list(arg(argc, argv, "--batches", "1,8"))) {
    const auto batch = static_cast<std::size_t>(b);
    const Inputs in = make_inputs(m, batch);
    for (int i = 0; i < 5; ++i) run_once(m, in, batch);  // warm-up: arena allocation, kernel selection
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
      const auto s = Clock::now();
      run_once(m, in, batch);
      ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());
    }
    const double p50 = pct(ms, 0.5);
    rows.push_back({{"batch", b}, {"p50_ms", p50}, {"p95_ms", pct(ms, 0.95)}, {"per_item_ms", p50 / b},
                    {"items_per_s", 1000.0 * b / p50}, {"iters", iters}});
    std::fprintf(stderr, "  batch %3d  p50 %8.2f ms  p95 %8.2f ms  %8.1f items/s\n", b, p50, pct(ms, 0.95), 1000.0 * b / p50);
  }
  return {{"mode", "latency"}, {"model", mc.path}, {"intra_op_threads", mc.intra_op_threads}, {"load_ms", load_ms},
          {"rss_before_load_mib", rss_before}, {"peak_rss_mib", peak_rss_mib()}, {"rows", rows}};
}

json threads(int argc, char** argv) {
  const std::string path = arg(argc, argv, "--model", "");
  const double seconds = std::stod(arg(argc, argv, "--seconds", "5"));
  std::vector<bool> spins{false};
  if (flag(argc, argv, "--spin")) spins.push_back(true);
  json rows = json::array();
  for (bool spin : spins) {
    for (int t : int_list(arg(argc, argv, "--intra", "1,2,4"))) {
      ModelConfig mc;
      mc.path = path;
      mc.intra_op_threads = t;
      mc.allow_spinning = spin;
      Model m(mc);
      const Inputs in = make_inputs(m, 1);
      for (int i = 0; i < 5; ++i) run_once(m, in, 1);
      for (int c : int_list(arg(argc, argv, "--callers", "1,2,4"))) {
        std::atomic<bool> go{false}, stop{false};
        std::vector<std::vector<double>> lat(static_cast<std::size_t>(c));
        std::vector<std::thread> callers;
        for (int k = 0; k < c; ++k) {
          callers.emplace_back([&, k] {
            while (!go.load()) std::this_thread::yield();
            while (!stop.load(std::memory_order_relaxed)) {
              const auto s = Clock::now();
              run_once(m, in, 1);
              lat[static_cast<std::size_t>(k)].push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());
            }
          });
        }
        const auto start = Clock::now();
        go = true;
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
        for (auto& th : callers) th.join();
        const double wall = std::chrono::duration<double>(Clock::now() - start).count();
        std::vector<double> all;
        for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
        const double tput = static_cast<double>(all.size()) / wall;
        rows.push_back({{"intra_op_threads", t}, {"callers", c}, {"threads_total", t * c}, {"allow_spinning", spin},
                        {"inferences", all.size()}, {"throughput_per_s", tput}, {"p50_ms", pct(all, 0.5)}, {"p95_ms", pct(all, 0.95)}});
        std::fprintf(stderr, "  spin=%d intra=%d callers=%d total=%2d  %8.1f inf/s  p50 %7.2f  p95 %7.2f ms\n", spin, t, c, t * c, tput,
                     pct(all, 0.5), pct(all, 0.95));
      }
    }
  }
  return {{"mode", "threads"}, {"model", path}, {"seconds_per_cell", seconds}, {"hardware_threads", std::thread::hardware_concurrency()},
          {"rows", rows}};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: edge_bench latency|threads --model M [...] --out file.json\n";
    return 2;
  }
  const std::string mode = argv[1];
  json out = mode == "latency" ? latency(argc, argv) : threads(argc, argv);
  const std::string path = arg(argc, argv, "--out", "");
  if (!path.empty()) std::ofstream(path) << out.dump(2) << "\n";
  else std::cout << out.dump(2) << "\n";
  return 0;
}
