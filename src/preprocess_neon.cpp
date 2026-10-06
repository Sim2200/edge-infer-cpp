// NEON preprocessing for aarch64 (NEON is mandatory in ARMv8-A, so no run-time check is needed).
// Same structure as the AVX2 path at 4 lanes: gather the four bilinear corners per channel, then
// two horizontal lerps, one vertical lerp and scale + bias with fused multiply-adds (vfmaq_f32).
#include <arm_neon.h>

#include <stdexcept>
#include <vector>

#include "edgeinfer/preprocess.hpp"

namespace edgeinfer {

namespace {
inline float32x4_t widen4(const std::uint8_t* p) {
  // 8 bytes loaded, low 4 used: u8 -> u16 -> u32 -> f32.
  return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p)))));
}

inline float32x4_t lerp4(float32x4_t a, float32x4_t b, float32x4_t w) {
  return vfmaq_f32(a, vsubq_f32(b, a), w);  // a + (b - a) * w
}
}  // namespace

void preprocess_neon(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out) {
  if (out.size() < static_cast<std::size_t>(3) * out_w * out_h) throw std::invalid_argument("output too small");
  std::vector<AxisMap> xs(static_cast<std::size_t>(out_w)), ys(static_cast<std::size_t>(out_h));
  build_axis_map(in.width, out_w, xs.data());
  build_axis_map(in.height, out_h, ys.data());
  std::vector<int> off0(static_cast<std::size_t>(out_w)), off1(static_cast<std::size_t>(out_w));
  std::vector<float> wx(static_cast<std::size_t>(out_w));
  for (int x = 0; x < out_w; ++x) {
    off0[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].i0 * 3;
    off1[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].i1 * 3;
    wx[static_cast<std::size_t>(x)] = xs[static_cast<std::size_t>(x)].w1;
  }
  const std::size_t plane = static_cast<std::size_t>(out_w) * out_h;
  float sscale[3], sbias[3];
  float32x4_t vscale[3], vbias[3];
  for (int c = 0; c < 3; ++c) {
    sscale[c] = 1.0f / (255.0f * norm.std[c]);
    sbias[c] = -norm.mean[c] / norm.std[c];
    vscale[c] = vdupq_n_f32(sscale[c]);
    vbias[c] = vdupq_n_f32(sbias[c]);
  }
  alignas(16) std::uint8_t g[4][3][8];  // bytes, widened 4 at a time (see the AVX2 path)
  for (int y = 0; y < out_h; ++y) {
    const AxisMap ym = ys[static_cast<std::size_t>(y)];
    const std::uint8_t* r0 = in.data + static_cast<std::size_t>(ym.i0) * in.stride;
    const std::uint8_t* r1 = in.data + static_cast<std::size_t>(ym.i1) * in.stride;
    const float32x4_t vwy = vdupq_n_f32(ym.w1);
    float* dst[3] = {out.data() + static_cast<std::size_t>(y) * out_w, out.data() + plane + static_cast<std::size_t>(y) * out_w,
                     out.data() + 2 * plane + static_cast<std::size_t>(y) * out_w};
    int x = 0;
    for (; x + 4 <= out_w; x += 4) {
      for (int l = 0; l < 4; ++l) {
        const std::size_t xi = static_cast<std::size_t>(x + l);
        for (int c = 0; c < 3; ++c) {
          g[0][c][l] = r0[off0[xi] + c];
          g[1][c][l] = r0[off1[xi] + c];
          g[2][c][l] = r1[off0[xi] + c];
          g[3][c][l] = r1[off1[xi] + c];
        }
      }
      const float32x4_t vwx = vld1q_f32(wx.data() + x);
      for (int c = 0; c < 3; ++c) {
        const float32x4_t top = lerp4(widen4(g[0][c]), widen4(g[1][c]), vwx);
        const float32x4_t bot = lerp4(widen4(g[2][c]), widen4(g[3][c]), vwx);
        const float32x4_t v = lerp4(top, bot, vwy);
        vst1q_f32(dst[c] + x, vfmaq_f32(vbias[c], v, vscale[c]));
      }
    }
    for (; x < out_w; ++x) {
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
