// edge_loadgen: an open-loop load generator for edge_server.
//
//   edge_loadgen --host 127.0.0.1 --port 8080 --rate 100 --seconds 20 --kind vision --out results/x.json
//
// Open loop: request i is *scheduled* at time i / rate (Poisson arrivals, seed 7) whether or not
// earlier requests have finished, and its latency is measured from the scheduled time, not from
// when a sender thread got round to it. A closed-loop client (send, wait, send) slows down when
// the server does and hides queueing ("coordinated omission"); this one does not.
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
std::string arg(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i) if (argv[i] == key) return argv[i + 1];
  return def;
}
double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(p * static_cast<double>(v.size() - 1))];
}
}  // namespace

int main(int argc, char** argv) {
  const std::string host = arg(argc, argv, "--host", "127.0.0.1");
  const int port = std::stoi(arg(argc, argv, "--port", "8080"));
  const double rate = std::stod(arg(argc, argv, "--rate", "50"));
  const double seconds = std::stod(arg(argc, argv, "--seconds", "20"));
  const std::string kind = arg(argc, argv, "--kind", "vision");
  const int senders = std::stoi(arg(argc, argv, "--senders", "128"));  // must stay below the server's --http-threads
  const int img_w = std::stoi(arg(argc, argv, "--width", "320")), img_h = std::stoi(arg(argc, argv, "--height", "240"));

  // Request body: a fixed synthetic image (or token sequence); content does not change compute cost.
  std::mt19937 rng(7);
  std::string body, path, ctype;
  if (kind == "vision") {
    body.resize(static_cast<std::size_t>(img_w) * img_h * 3);
    for (auto& c : body) c = static_cast<char>(rng() & 0xFF);
    path = "/v1/infer?w=" + std::to_string(img_w) + "&h=" + std::to_string(img_h);
    ctype = "application/octet-stream";
  } else {
    const int len = std::stoi(arg(argc, argv, "--tokens", "128"));
    json ids = json::array({101});
    std::uniform_int_distribution<int> tok(1000, 29000);
    for (int i = 0; i < len - 2; ++i) ids.push_back(tok(rng));
    ids.push_back(102);
    body = json{{"input_ids", ids}}.dump();
    path = "/v1/infer";
    ctype = "application/json";
  }

  // Poisson schedule.
  std::exponential_distribution<double> gap(rate);
  std::vector<double> at;
  for (double t = gap(rng); t < seconds; t += gap(rng)) at.push_back(t);
  const std::size_t n = at.size();

  std::vector<double> lat(n, -1.0);
  std::vector<int> status(n, 0), batch(n, 0);
  std::atomic<std::size_t> next{0};
  const auto t0 = Clock::now() + std::chrono::milliseconds(200);
  std::vector<std::thread> pool;
  for (int s = 0; s < senders; ++s) {
    pool.emplace_back([&] {
      httplib::Client cli(host, port);
      cli.set_keep_alive(true);
      cli.set_read_timeout(30, 0);
      for (;;) {
        const std::size_t i = next.fetch_add(1);
        if (i >= n) return;
        const auto due = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(at[i]));
        std::this_thread::sleep_until(due);
        auto res = cli.Post(path, body, ctype);
        lat[i] = std::chrono::duration<double, std::milli>(Clock::now() - due).count();
        status[i] = res ? res->status : -1;
        if (res && res->status == 200) {
          try {
            batch[i] = json::parse(res->body).value("batch_size", 0);
          } catch (...) {
          }
        }
      }
    });
  }
  for (auto& th : pool) th.join();
  const double wall = std::chrono::duration<double>(Clock::now() - t0).count();

  std::vector<double> ok_lat;
  std::size_t ok = 0, rejected = 0, failed = 0;
  double batch_sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (status[i] == 200) {
      ++ok;
      ok_lat.push_back(lat[i]);
      batch_sum += batch[i];
    } else if (status[i] == 503) {
      ++rejected;
    } else {
      ++failed;
    }
  }
  json out{{"offered_rps", rate}, {"seconds", seconds}, {"scheduled", n}, {"ok", ok}, {"rejected_503", rejected}, {"failed", failed},
           {"goodput_rps", static_cast<double>(ok) / wall}, {"p50_ms", pct(ok_lat, 0.5)}, {"p95_ms", pct(ok_lat, 0.95)},
           {"p99_ms", pct(ok_lat, 0.99)}, {"mean_batch_size", ok ? batch_sum / static_cast<double>(ok) : 0.0}, {"kind", kind}};
  std::fprintf(stderr, "offered %.0f rps: ok %zu, 503 %zu, failed %zu, goodput %.1f rps, p50 %.1f p95 %.1f p99 %.1f ms, mean batch %.2f\n", rate, ok,
               rejected, failed, out["goodput_rps"].get<double>(), out["p50_ms"].get<double>(), out["p95_ms"].get<double>(),
               out["p99_ms"].get<double>(), out["mean_batch_size"].get<double>());
  const std::string outp = arg(argc, argv, "--out", "");
  if (!outp.empty()) std::ofstream(outp) << out.dump(2) << "\n";
  return 0;
}
