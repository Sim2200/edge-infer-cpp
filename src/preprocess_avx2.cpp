// AVX2 + FMA preprocessing. Compiled with -mavx2 -mfma in its own translation unit only; the
// dispatcher calls it after checking the CPU at run time.
//
// Strategy: 8 output pixels per iteration along a row.
//
// Bilinear sampling needs four source pixels per output pixel (two columns x two rows), and the
// source is RGB interleaved, so it is a gather by nature. Instead of loading 12 bytes one at a
// time, each corner is fetched with ONE `vpgatherdd` that loads the 4 bytes starting at that
// pixel: R, G, B and one byte of the next pixel. A per-lane variable shift (`vpsrlvd`) and a mask
// then extract each channel. That is 4 gathers per 8 output pixels instead of 96 scalar byte loads.
//
// Bounds: a 4-byte load at the last pixel of a row would read one byte past the row. For those
// columns the load starts one byte earlier (at the previous pixel's blue) and the shift skips it;
// `go` (gather offset) and `sh` (bit shift of the red byte inside the loaded word) encode this per
// column, computed once per image. Rows narrower than 2 pixels fall back to the scalar path.
//
// The arithmetic (two horizontal lerps, one vertical, scale + bias) is 8-wide FMA and mirrors the
// scalar reference operation for operation; FMA's single rounding differs from mul + add by at
// most a few ULP, so the tests compare with a 1e-5 tolerance.
#include <immintrin.h>

#include <stdexcept>
#include <vector>

#include "edgeinfer/preprocess.hpp"

namespace edgeinfer {

namespace {
inline __m256 lerp8(__m256 a, __m256 b, __m256 w) {
  return _mm256_fmadd_ps(_mm256_sub_ps(b, a), w, a);  // a + (b - a) * w
}

// Channel c of the 8 gathered words, as floats.
inline __m256 channel(__m256i words, __m256i shift, int c, __m256i mask) {
  const __m256i s = _mm256_add_epi32(shift, _mm256_set1_epi32(8 * c));
  return _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(words, s), mask));
}
}  // namespace

void preprocess_avx2(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out) {
  if (out.size() < static_cast<std::size_t>(3) * out_w * out_h) throw std::invalid_argument("output too small");
  if (in.width < 2) {
    preprocess_scalar(in, out_w, out_h, norm, out);
    return;
  }
  std::vector<AxisMap> xs(static_cast<std::size_t>(out_w)), ys(static_cast<std::size_t>(out_h));
  build_axis_map(in.width, out_w, xs.data());
  build_axis_map(in.height, out_h, ys.data());

  const int row_bytes = in.width * 3;
  const auto n = static_cast<std::size_t>(out_w);
  std::vector<int> go0(n), sh0(n), go1(n), sh1(n);  // gather byte offset and shift, per column
  std::vector<float> wx(n);
  auto place = [row_bytes](int pixel, int& go, int& sh) {
    const int off = pixel * 3;
    go = off + 4 <= row_bytes ? off : row_bytes - 4;  // never read past the row
    sh = (off - go) * 8;
  };
  for (std::size_t x = 0; x < n; ++x) {
    place(xs[x].i0, go0[x], sh0[x]);
    place(xs[x].i1, go1[x], sh1[x]);
    wx[x] = xs[x].w1;
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
  const __m256i mask = _mm256_set1_epi32(0xFF);

  for (int y = 0; y < out_h; ++y) {
    const AxisMap ym = ys[static_cast<std::size_t>(y)];
    const auto* r0 = reinterpret_cast<const int*>(in.data + static_cast<std::size_t>(ym.i0) * in.stride);
    const auto* r1 = reinterpret_cast<const int*>(in.data + static_cast<std::size_t>(ym.i1) * in.stride);
    const __m256 vwy = _mm256_set1_ps(ym.w1);
    float* dst[3] = {out.data() + static_cast<std::size_t>(y) * out_w, out.data() + plane + static_cast<std::size_t>(y) * out_w,
                     out.data() + 2 * plane + static_cast<std::size_t>(y) * out_w};
    int x = 0;
    for (; x + 8 <= out_w; x += 8) {
      const __m256i i0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(go0.data() + x));
      const __m256i i1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(go1.data() + x));
      const __m256i s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sh0.data() + x));
      const __m256i s1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sh1.data() + x));
      // scale = 1: the indices are byte offsets.
      const __m256i w00 = _mm256_i32gather_epi32(r0, i0, 1);
      const __m256i w01 = _mm256_i32gather_epi32(r0, i1, 1);
      const __m256i w10 = _mm256_i32gather_epi32(r1, i0, 1);
      const __m256i w11 = _mm256_i32gather_epi32(r1, i1, 1);
      const __m256 vwx = _mm256_loadu_ps(wx.data() + x);
      for (int c = 0; c < 3; ++c) {
        const __m256 top = lerp8(channel(w00, s0, c, mask), channel(w01, s1, c, mask), vwx);
        const __m256 bot = lerp8(channel(w10, s0, c, mask), channel(w11, s1, c, mask), vwx);
        const __m256 v = lerp8(top, bot, vwy);
        _mm256_storeu_ps(dst[c] + x, _mm256_fmadd_ps(v, vscale[c], vbias[c]));
      }
    }
    const auto* b0 = reinterpret_cast<const std::uint8_t*>(r0);
    const auto* b1 = reinterpret_cast<const std::uint8_t*>(r1);
    for (; x < out_w; ++x) {  // tail: same formula, scalar, fmaf to match the vector rounding
      const std::size_t xi = static_cast<std::size_t>(x);
      const int o0 = xs[xi].i0 * 3, o1 = xs[xi].i1 * 3;
      for (int c = 0; c < 3; ++c) {
        const float p00 = b0[o0 + c], p01 = b0[o1 + c];
        const float p10 = b1[o0 + c], p11 = b1[o1 + c];
        const float top = __builtin_fmaf(p01 - p00, wx[xi], p00);
        const float bot = __builtin_fmaf(p11 - p10, wx[xi], p10);
        const float v = __builtin_fmaf(bot - top, ym.w1, top);
        dst[c][x] = __builtin_fmaf(v, sscale[c], sbias[c]);
      }
    }
  }
}

}  // namespace edgeinfer
