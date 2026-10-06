#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "edgeinfer/preprocess.hpp"

using namespace edgeinfer;

TEST(PreprocessTest, AxisMapIdentity) {
  AxisMap maps[5];
  build_axis_map(5, 5, maps);

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(maps[i].i0, i);
    // i1 is clamped: min(i0+1, in_size-1), so for last element i1==i0
    if (i < 4) {
      EXPECT_EQ(maps[i].i1, i + 1);
    } else {
      EXPECT_EQ(maps[i].i1, 4);  // clamped
    }
    EXPECT_FLOAT_EQ(maps[i].w1, 0.0f);
  }
}

TEST(PreprocessTest, AxisMapDownscale2x) {
  AxisMap maps[4];
  build_axis_map(8, 4, maps);

  for (int o = 0; o < 4; ++o) {
    EXPECT_EQ(maps[o].i0, 2 * o);
    EXPECT_FLOAT_EQ(maps[o].w1, 0.5f);
  }
}

TEST(PreprocessTest, ScalarConstantImage) {
  // Just verify that constant input produces finite output
  const int w = 32, h = 32;
  std::vector<uint8_t> image(3 * w * h);
  std::fill(image.begin(), image.end(), 128);

  std::vector<float> out(3 * 224 * 224);
  ImageView view{image.data(), w, h, 3 * w};
  Normalize norm;

  preprocess_scalar(view, 224, 224, norm, out);

  // Just verify all outputs are finite
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_TRUE(std::isfinite(out[i])) << "Non-finite value at index " << i;
  }
}

TEST(PreprocessTest, SimdMatchesScalar) {
  std::vector<std::pair<int, int>> sizes = {
      {640, 480}, {224, 224}, {33, 17}, {1000, 3}
  };
  std::vector<std::pair<int, int>> outputs = {
      {224, 224}, {224, 224}, {224, 224}, {7, 5}
  };

  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, 255);

  Normalize norm;

  for (size_t test = 0; test < sizes.size(); ++test) {
    int in_w = sizes[test].first, in_h = sizes[test].second;
    int out_w = outputs[test].first, out_h = outputs[test].second;

    std::vector<uint8_t> image(3 * in_w * in_h);
    for (auto& b : image) b = static_cast<uint8_t>(dis(gen));

    std::vector<float> out_scalar(3 * out_w * out_h);
    std::vector<float> out_simd(3 * out_w * out_h);

    ImageView view{image.data(), in_w, in_h, 3 * in_w};

    preprocess_scalar(view, out_w, out_h, norm, out_scalar);

#ifdef EDGEINFER_HAVE_AVX2
    preprocess_avx2(view, out_w, out_h, norm, out_simd);
#elif defined(EDGEINFER_HAVE_NEON)
    preprocess_neon(view, out_w, out_h, norm, out_simd);
#else
    preprocess(view, out_w, out_h, norm, out_simd);
#endif

    float max_error = 0.0f;
    for (size_t i = 0; i < out_scalar.size(); ++i) {
      float error = std::abs(out_scalar[i] - out_simd[i]);
      max_error = std::max(max_error, error);
      EXPECT_LE(error, 1e-5f) << "Mismatch at index " << i;
    }

    RecordProperty("max_abs_error_test_" + std::to_string(test),
                   std::to_string(max_error));
  }
}

TEST(PreprocessTest, SelectedPath) {
  auto path = selected_path();
  EXPECT_TRUE(path == "avx2" || path == "neon" || path == "scalar");
}

TEST(PreprocessTest, LargeStride) {
  const int w = 8, h = 8;
  const int stride = 3 * w + 100;  // Larger than 3*width

  std::vector<uint8_t> image(stride * h + 100);
  std::fill(image.begin(), image.end(), 128);

  std::vector<float> out_padded(3 * 224 * 224);
  std::vector<float> out_unpadded(3 * 224 * 224);

  // With padding (garbage bytes in the stride gap)
  std::mt19937 gen(42);
  for (size_t i = 3 * w; i < stride; ++i) {
    image[i] = static_cast<uint8_t>(gen() % 256);
  }

  ImageView view_padded{image.data(), w, h, stride};
  Normalize norm;
  preprocess_scalar(view_padded, 224, 224, norm, out_padded);

  // Without padding (but should give same result since only original pixels used)
  std::vector<uint8_t> compact(3 * w * h);
  for (int r = 0; r < h; ++r) {
    std::memcpy(compact.data() + r * 3 * w,
                image.data() + r * stride,
                3 * w);
  }

  ImageView view_compact{compact.data(), w, h, 3 * w};
  preprocess_scalar(view_compact, 224, 224, norm, out_unpadded);

  for (size_t i = 0; i < out_padded.size(); ++i) {
    EXPECT_FLOAT_EQ(out_padded[i], out_unpadded[i]);
  }
}

TEST(PreprocessTest, OutputTooSmall) {
  const int w = 10, h = 10;
  std::vector<uint8_t> image(3 * w * h);
  std::vector<float> out(10);  // Too small

  ImageView view{image.data(), w, h, 3 * w};
  Normalize norm;

  EXPECT_THROW(preprocess_scalar(view, 224, 224, norm, out),
               std::invalid_argument);
}
