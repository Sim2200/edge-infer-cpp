#pragma once
// A thin RAII wrapper over an ONNX Runtime session. It reads the model's input signature to decide
// whether it is a vision model (one float NCHW input) or a text model (int64 input_ids and
// attention_mask), and exposes one batched `run` per kind. The session is created once and is
// safe to call from several threads (OrtSession::Run is thread-safe).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Ort {
struct Session;
struct Env;
}  // namespace Ort

namespace edgeinfer {

struct ModelConfig {
  std::string path;
  int intra_op_threads = 1;   // threads ORT uses inside one operator (e.g. a conv)
  int inter_op_threads = 1;   // threads for independent graph branches (parallel execution mode only)
  std::string graph_opt = "all";  // "none" | "basic" | "extended" | "all"
  bool allow_spinning = false;    // ORT's spin-wait between ops; off by default (see docs/architecture.md)
};

enum class ModelKind { Vision, Text };

class Model {
 public:
  explicit Model(const ModelConfig& cfg);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  ModelKind kind() const { return kind_; }
  const std::string& path() const { return cfg_.path; }
  int input_size() const { return input_hw_; }       // vision: H == W (224 for the models here)
  std::size_t output_dim() const { return out_dim_; }  // classes per row

  // Vision: `chw` holds n * 3 * H * W floats. Returns n * output_dim logits.
  std::vector<float> run_vision(const float* chw, std::size_t n) const;
  // Text: n rows of `seq` token ids and attention mask. Returns n * output_dim logits.
  std::vector<float> run_text(const std::int64_t* ids, const std::int64_t* mask, std::size_t n, std::size_t seq) const;

 private:
  ModelConfig cfg_;
  std::unique_ptr<Ort::Session> session_;
  ModelKind kind_ = ModelKind::Vision;
  std::vector<std::string> input_names_;
  std::string output_name_;
  int input_hw_ = 0;
  std::size_t out_dim_ = 0;
};

// Peak resident set size of this process in MiB (getrusage ru_maxrss).
double peak_rss_mib();

}  // namespace edgeinfer
