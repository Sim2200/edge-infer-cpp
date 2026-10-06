// edge_server: one model per process, served over HTTP with dynamic micro-batching.
//
//   edge_server --model models/resnet50_v2_int8/model.onnx --port 8080 --max-batch 8 --max-wait-us 2000
//
// POST /v1/infer
//   vision: body = raw RGB8 bytes, query ?w=<width>&h=<height>; preprocessing (resize + normalise,
//           SIMD) runs on the HTTP thread, so it overlaps with inference of earlier requests.
//   text:   body = {"input_ids": [101, ..., 102]} (attention mask = all ones)
//   ?logits=1 adds the full logits to the response (used by the parity test).
// GET /metrics   Prometheus text format
// GET /health    {"status":"ok", ...}
#include <httplib.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "edgeinfer/batcher.hpp"
#include "edgeinfer/metrics.hpp"
#include "edgeinfer/model.hpp"
#include "edgeinfer/preprocess.hpp"

using namespace edgeinfer;
using json = nlohmann::json;

namespace {

struct Args {
  std::string model;
  std::string host = "0.0.0.0";
  int port = 8080;
  BatcherConfig batch;
  int intra = 1;
  int http_threads = 256;  // >= concurrent keep-alive clients: httplib pins one thread per open connection
};

long long arg_int(const char* v) {
  long long out = 0;
  const char* end = v + std::char_traits<char>::length(v);
  if (std::from_chars(v, end, out).ec != std::errc{}) throw std::invalid_argument(std::string("not an integer: ") + v);
  return out;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
      return argv[++i];
    };
    if (k == "--model") a.model = next();
    else if (k == "--host") a.host = next();
    else if (k == "--port") a.port = static_cast<int>(arg_int(next()));
    else if (k == "--max-batch") a.batch.max_batch = static_cast<std::size_t>(arg_int(next()));
    else if (k == "--max-wait-us") a.batch.max_wait = std::chrono::microseconds(arg_int(next()));
    else if (k == "--workers") a.batch.workers = static_cast<std::size_t>(arg_int(next()));
    else if (k == "--queue") a.batch.queue_capacity = static_cast<std::size_t>(arg_int(next()));
    else if (k == "--intra") a.intra = static_cast<int>(arg_int(next()));
    else if (k == "--http-threads") a.http_threads = static_cast<int>(arg_int(next()));
    else throw std::invalid_argument("unknown argument " + k);
  }
  if (a.model.empty()) throw std::invalid_argument("--model is required");
  return a;
}

Batcher::RunBatch make_runner(const Model& model) {
  if (model.kind() == ModelKind::Vision) {
    return [&model](const std::vector<Payload*>& batch) {
      const std::size_t per = batch.front()->image.size();
      std::vector<float> buf(per * batch.size());  // concatenate into one NCHW tensor
      for (std::size_t i = 0; i < batch.size(); ++i) std::copy(batch[i]->image.begin(), batch[i]->image.end(), buf.begin() + static_cast<std::ptrdiff_t>(i * per));
      return model.run_vision(buf.data(), batch.size());
    };
  }
  return [&model](const std::vector<Payload*>& batch) {
    std::size_t seq = 0;
    for (const auto* p : batch) seq = std::max(seq, p->tokens.size());
    std::vector<std::int64_t> ids(batch.size() * seq, 0), mask(batch.size() * seq, 0);  // pad right with 0 / mask 0
    for (std::size_t i = 0; i < batch.size(); ++i) {
      std::copy(batch[i]->tokens.begin(), batch[i]->tokens.end(), ids.begin() + static_cast<std::ptrdiff_t>(i * seq));
      std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(i * seq), batch[i]->tokens.size(), 1);
    }
    return model.run_text(ids.data(), mask.data(), batch.size(), seq);
  };
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  ModelConfig mc;
  mc.path = args.model;
  mc.intra_op_threads = args.intra;
  Model model(mc);
  ServerMetrics metrics;
  Batcher batcher(args.batch, model.output_dim(), make_runner(model), &metrics);
  const int hw = model.input_size();

  httplib::Server svr;
  svr.new_task_queue = [n = args.http_threads] { return new httplib::ThreadPool(static_cast<std::size_t>(n)); };
  svr.set_keep_alive_max_count(100000);
  svr.set_keep_alive_timeout(30);

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    json j{{"status", "ok"}, {"model", model.path()}, {"kind", model.kind() == ModelKind::Vision ? "vision" : "text"},
           {"preprocess", std::string(selected_path())}, {"max_batch", args.batch.max_batch},
           {"max_wait_us", args.batch.max_wait.count()}, {"workers", args.batch.workers}, {"intra_op_threads", args.intra},
           {"peak_rss_mib", peak_rss_mib()}};
    res.set_content(j.dump(), "application/json");
  });

  svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(metrics.render(), "text/plain; version=0.0.4");
  });

  svr.Post("/v1/infer", [&](const httplib::Request& req, httplib::Response& res) {
    Payload p;
    try {
      if (model.kind() == ModelKind::Vision) {
        const int w = static_cast<int>(arg_int(req.get_param_value("w").c_str()));
        const int h = static_cast<int>(arg_int(req.get_param_value("h").c_str()));
        if (w <= 0 || h <= 0 || req.body.size() != static_cast<std::size_t>(w) * h * 3) throw std::invalid_argument("body must be w*h*3 RGB bytes");
        p.image.resize(static_cast<std::size_t>(3) * hw * hw);
        const ImageView img{reinterpret_cast<const std::uint8_t*>(req.body.data()), w, h, w * 3};
        preprocess(img, hw, hw, Normalize{}, p.image);
      } else {
        p.tokens = json::parse(req.body).at("input_ids").get<std::vector<std::int64_t>>();
        if (p.tokens.empty() || p.tokens.size() > 512) throw std::invalid_argument("input_ids must have 1..512 tokens");
      }
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(json{{"error", e.what()}}.dump(), "application/json");
      return;
    }
    auto fut = batcher.submit(std::move(p));
    if (!fut) {
      res.status = 503;
      res.set_content(R"({"error":"queue full"})", "application/json");
      return;
    }
    try {
      Result r = fut->get();
      const auto best = std::max_element(r.logits.begin(), r.logits.end());
      json j{{"class", best - r.logits.begin()}, {"score", *best}, {"batch_size", r.batch_size},
             {"queue_wait_ms", r.queue_wait_ms}, {"infer_ms", r.infer_ms}};
      if (req.get_param_value("logits") == "1") j["logits"] = r.logits;
      res.set_content(j.dump(), "application/json");
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
  });

  std::fprintf(stderr, "edge_server: %s (%s), max_batch=%zu max_wait=%lldus workers=%zu intra=%d preprocess=%s on %s:%d\n",
               model.path().c_str(), model.kind() == ModelKind::Vision ? "vision" : "text", args.batch.max_batch,
               static_cast<long long>(args.batch.max_wait.count()), args.batch.workers, args.intra,
               std::string(selected_path()).c_str(), args.host.c_str(), args.port);
  return svr.listen(args.host, args.port) ? 0 : 1;
}
