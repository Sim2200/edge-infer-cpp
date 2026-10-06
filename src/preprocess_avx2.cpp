// AVX2 + FMA preprocessing. Compiled with -mavx2 -mfma in its own translation unit only; the
// dispatcher calls it after checking the CPU at run time.
//
// Strategy: 8 output pixels per iteration along a row. The source pixels are gathered as bytes
// into small per-channel arrays (bilinear sampling is a gather by nature, and RGB interleave makes
// it a strided one), widened to float, and then all arithmetic (two horizontal lerps, one vertical
// lerp, scale + bias) runs 8-wide with FMA. The y-direction weights and the x-direction index map
// are computed once per image, not per pixel. The arithmetic mirrors the scalar reference operation
// for operation (same lerp form, same FMA order), which is what makes a bit-exact comparison
// possible except where FMA's single rounding differs from mul + add: the tests allow 1e-5.
#include <immintrin.h>

#include <stdexcept>
#include <vector>

#include "edgeinfer/preprocess.hpp"

namespace edgeinfer {

namespace {
inline __m256 lerp8(__m256 a, __m256 b, __m256 w) {
  // a + (b - a) * w, as one FMA
  return _mm256_fmadd_ps(_mm256_sub_ps(b, a), w, a);
}
}  // namespace

void preprocess_avx2(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out) {
  if (out.size() < static_cast<std::size_t>(3) * out_w * out_h) throw std::invalid_argument("output too small");
  std::vector<AxisMap> xs(static_cast<std::size_t>(out_w)), ys(static_cast<std::size_t>(out_h));
  build_axis_map(in.width, out_w, xs.data());
  build_axis_map(in.height, out_h, ys.data());
  // Byte offsets of the left/right source pixel for every output column, and the x weights.
  std::vector<int> off0(static_cast<std::size_t>(out_w)), off1(static_cast<std::size_t>(out_w));
  std::vector<float> wx(static_cast<std::size_t>(out_w));
  for (int x = 0; x < out_w; ++x) {
    off0[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].i0 * 3;
    off1[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].i1 * 3;
    wx[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].w1;
  }
  const std::size_t plane = static_cast<std::size_t>(out_w) * out_h;
  __m256 vscale[3], vbias[3];
  float sscale[3], sbias[3];
  for (int c = 0; c < 3; ++c) {
    sscale[c] = 1.0f / (255.0f * norm.std[c]);
    sbias[c] = -norm.mean[c] / norm.std[c];
    vscale[c] = _mm256_set1_ps(sscale[c]);
    vbias[c] = _mm256_set1_ps(sbias[c]);
  }

  alignas(32) float g[4][3][8];  // [corner][channel][lane]: p00 p01 p10 p11
  for (int y = 0; y < out_h; ++y) {
    const AxisMap ym = ys[static_cast<std::size_t>(y)];
    const std::uint8_t* r0 = in.data + static_cast<std::size_t>(ym.i0) * in.stride;
    const std::uint8_t* r1 = in.data + static_cast<std::size_t>(ym.i1) * in.stride;
    const __m256 vwy = _mm256_set1_ps(ym.w1);
    float* dst[3] = {out.data() + static_cast<std::size_t>(y) * out_w, out.data() + plane + static_cast<std::size_t>(y) * out_w,
                     out.data() + 2 * plane + static_cast<std::size_t>(y) * out_w};
    int x = 0;
    for (; x + 8 <= out_w; x += 8) {
      for (int l = 0; l < 8; ++l) {
        const std::size_t xi = static_cast<std::size_t>(x + l);
        const std::uint8_t* a0 = r0 + off0[xi];
        const std::uint8_t* a1 = r0 + off1[xi];
        const std::uint8_t* b0 = r1 + off0[xi];
        const std::uint8_t* b1 = r1 + off1[xi];
        for (int c = 0; c < 3; ++c) {
          g[0][c][l] = a0[c];
          g[1][c][l] = a1[c];
          g[2][c][l] = b0[c];
          g[3][c][l] = b1[c];
        }
      }
      const __m256 vwx = _mm256_loadu_ps(wx.data() + x);
      for (int c = 0; c < 3; ++c) {
        const __m256 top = lerp8(_mm256_load_ps(g[0][c]), _mm256_load_ps(g[1][c]), vwx);
        const __m256 bot = lerp8(_mm256_load_ps(g[2][c]), _mm256_load_ps(g[3][c]), vwx);
        const __m256 v = lerp8(top, bot, vwy);
        _mm256_storeu_ps(dst[c] + x, _mm256_fmadd_ps(v, vscale[c], vbias[c]));
      }
    }
    for (; x < out_w; ++x) {  // tail: same formula, scalar, using fmaf to match the vector rounding
      const std::size_t xi = static_cast<std::size_t>(x);
      for (int c = 0; c < 3; ++c) {
        const float p00 = r0[off0[xi] + c], p01 = r0[off1[xi] + c];
        const float p10 = r1[off0[xi] + c], p11 = r1[off1[xi] + c];
        const float top = __builtin_fmaf(p01 - p00, wx[xi], p00);
        const float bot = __builtin_fmaf(p11 - p10, wx[xi], p10);
        const float v = __builtin_fmaf(bot - top, ym.w1, top);
        dst[c][x] = __builtin_fmaf(v, sscale[c], sbias[c]);
      }
    }
  }
}

}  // namespace edgeinfer
