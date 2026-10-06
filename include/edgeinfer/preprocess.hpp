#pragma once
// Image preprocessing for the vision models: RGB8 interleaved (HWC) -> float32 planar (CHW),
// bilinear resize to the model's input size, then per-channel (x / 255 - mean) / std.
//
// This is the step every image request pays before inference. Three implementations share one
// signature: a scalar reference (the definition of correct), AVX2 on x86-64 and NEON on aarch64.
// `preprocess()` picks the fastest one the CPU supports at run time.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace edgeinfer {

struct ImageView {
  const std::uint8_t* data;  // RGB, row-major, 3 bytes per pixel
  int width;
  int height;
  int stride;  // bytes per row (>= 3 * width)
};

struct Normalize {
  float mean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet, the models in models/ expect these
  float std[3] = {0.229f, 0.224f, 0.225f};
};

// `out` must hold 3 * out_h * out_w floats, written as [C][H][W].
void preprocess_scalar(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out);
#if defined(EDGEINFER_HAVE_AVX2)
void preprocess_avx2(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out);
#endif
#if defined(EDGEINFER_HAVE_NEON)
void preprocess_neon(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out);
#endif

// Dispatches to the best available path; `selected_path()` reports which one ("avx2", "neon", "scalar").
void preprocess(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out);
std::string_view selected_path();

// Bilinear sampling geometry shared by every path, so all three compute identical source
// coordinates (align_corners = false, the PyTorch / OpenCV INTER_LINEAR convention).
struct AxisMap {
  int i0;     // left/top source index
  int i1;     // right/bottom source index (clamped)
  float w1;   // weight of i1; weight of i0 is 1 - w1
};
void build_axis_map(int in_size, int out_size, AxisMap* map);

}  // namespace edgeinfer
