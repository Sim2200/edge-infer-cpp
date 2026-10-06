// Scalar reference for preprocessing. Every SIMD path is tested against this one.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "edgeinfer/preprocess.hpp"

namespace edgeinfer {

void build_axis_map(int in_size, int out_size, AxisMap* map) {
  const float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
  for (int o = 0; o < out_size; ++o) {
    // Centre of output pixel o, mapped back to source coordinates (align_corners = false).
    float src = (static_cast<float>(o) + 0.5f) * scale - 0.5f;
    src = std::max(src, 0.0f);
    const int i0 = std::min(static_cast<int>(src), in_size - 1);
    const int i1 = std::min(i0 + 1, in_size - 1);
    map[o] = {i0, i1, src - static_cast<float>(i0)};
  }
}

void preprocess_scalar(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out) {
  if (out.size() < static_cast<std::size_t>(3) * out_w * out_h) throw std::invalid_argument("output too small");
  std::vector<AxisMap> xs(static_cast<std::size_t>(out_w)), ys(static_cast<std::size_t>(out_h));
  build_axis_map(in.width, out_w, xs.data());
  build_axis_map(in.height, out_h, ys.data());
  const std::size_t plane = static_cast<std::size_t>(out_w) * out_h;
  float scale[3], bias[3];
  for (int c = 0; c < 3; ++c) {
    // (x / 255 - mean) / std  ==  x * (1 / (255 * std)) + (-mean / std): one fused multiply-add.
    scale[c] = 1.0f / (255.0f * norm.std[c]);
    bias[c] = -norm.mean[c] / norm.std[c];
  }
  for (int y = 0; y < out_h; ++y) {
    const AxisMap ym = ys[static_cast<std::size_t>(y)];
    const std::uint8_t* r0 = in.data + static_cast<std::size_t>(ym.i0) * in.stride;
    const std::uint8_t* r1 = in.data + static_cast<std::size_t>(ym.i1) * in.stride;
    for (int x = 0; x < out_w; ++x) {
      const AxisMap xm = xs[static_cast<std::size_t>(x)];
      for (int c = 0; c < 3; ++c) {
        const float p00 = r0[xm.i0 * 3 + c], p01 = r0[xm.i1 * 3 + c];
        const float p10 = r1[xm.i0 * 3 + c], p11 = r1[xm.i1 * 3 + c];
        const float top = p00 + (p01 - p00) * xm.w1;
        const float bot = p10 + (p11 - p10) * xm.w1;
        const float v = top + (bot - top) * ym.w1;
        out[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * out_w + x] = v * scale[c] + bias[c];
      }
    }
  }
}

}  // namespace edgeinfer
