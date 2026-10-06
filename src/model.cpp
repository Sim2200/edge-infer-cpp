#include "edgeinfer/model.hpp"

#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>

#include <array>
#include <stdexcept>

namespace edgeinfer {

namespace {
// One Ort::Env per process (it owns the logging and the global thread pools, if any).
Ort::Env& env() {
  static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "edge-infer");
  return e;
}

GraphOptimizationLevel opt_level(const std::string& s) {
  if (s == "none") return GraphOptimizationLevel::ORT_DISABLE_ALL;
  if (s == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (s == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  return GraphOptimizationLevel::ORT_ENABLE_ALL;
}
}  // namespace

Model::Model(const ModelConfig& cfg) : cfg_(cfg) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(cfg.intra_op_threads);
  so.SetInterOpNumThreads(cfg.inter_op_threads);
  so.SetGraphOptimizationLevel(opt_level(cfg.graph_opt));
  // Spinning keeps idle ORT threads busy-waiting for the next op. Inside a CPU-limited container
  // that burns quota the other threads need (the CFS-throttling finding from my Python gateway);
  // it stays off unless a benchmark turns it on to measure the difference.
  so.AddConfigEntry("session.intra_op.allow_spinning", cfg.allow_spinning ? "1" : "0");
  so.AddConfigEntry("session.inter_op.allow_spinning", cfg.allow_spinning ? "1" : "0");
  session_ = std::make_unique<Ort::Session>(env(), cfg.path.c_str(), so);

  Ort::AllocatorWithDefaultOptions alloc;
  const std::size_t n_in = session_->GetInputCount();
  for (std::size_t i = 0; i < n_in; ++i) input_names_.emplace_back(session_->GetInputNameAllocated(i, alloc).get());
  output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

  // The TypeInfo must outlive the shape view: GetTensorTypeAndShapeInfo() returns a non-owning
  // view into it, and calling it on a temporary leaves the view dangling (a real bug found here).
  const Ort::TypeInfo in0_info = session_->GetInputTypeInfo(0);
  const auto in0 = in0_info.GetTensorTypeAndShapeInfo();
  if (in0.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    kind_ = ModelKind::Text;
    if (n_in != 2) throw std::runtime_error("text model must take input_ids and attention_mask");
  } else {
    kind_ = ModelKind::Vision;
    const auto shape = in0.GetShape();  // [batch, 3, H, W]
    if (shape.size() != 4 || shape[1] != 3) throw std::runtime_error("vision model must take [N,3,H,W]");
    input_hw_ = static_cast<int>(shape[2]);
  }
  const Ort::TypeInfo out_info = session_->GetOutputTypeInfo(0);
  const auto out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
  out_dim_ = static_cast<std::size_t>(out_shape.back());
}

Model::~Model() = default;

std::vector<float> Model::run_vision(const float* chw, std::size_t n) const {
  const auto hw = static_cast<std::int64_t>(input_hw_);
  const std::array<std::int64_t, 4> shape{static_cast<std::int64_t>(n), 3, hw, hw};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // CreateTensor wraps the caller's buffer without copying; ORT only reads it.
  Ort::Value input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(chw), n * 3 * static_cast<std::size_t>(hw * hw),
                                                     shape.data(), shape.size());
  const char* in_names[] = {input_names_[0].c_str()};
  const char* out_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  const float* logits = outputs[0].GetTensorData<float>();
  return {logits, logits + n * out_dim_};
}

std::vector<float> Model::run_text(const std::int64_t* ids, const std::int64_t* mask, std::size_t n, std::size_t seq) const {
  const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(seq)};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<std::int64_t>(mem, const_cast<std::int64_t*>(ids), n * seq, shape.data(), shape.size()),
      Ort::Value::CreateTensor<std::int64_t>(mem, const_cast<std::int64_t*>(mask), n * seq, shape.data(), shape.size())};
  // Bind by name: the model's input order is not guaranteed to be ids-then-mask.
  std::array<const char*, 2> in_names{};
  std::array<Ort::Value, 2> ordered{Ort::Value{nullptr}, Ort::Value{nullptr}};
  for (std::size_t i = 0; i < 2; ++i) {
    in_names[i] = input_names_[i].c_str();
    ordered[i] = std::move(inputs[input_names_[i] == "attention_mask" ? 1 : 0]);
  }
  const char* out_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), ordered.data(), 2, out_names, 1);
  const float* logits = outputs[0].GetTensorData<float>();
  return {logits, logits + n * out_dim_};
}

double peak_rss_mib() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return static_cast<double>(ru.ru_maxrss) / 1024.0;  // Linux reports KiB
}

}  // namespace edgeinfer
