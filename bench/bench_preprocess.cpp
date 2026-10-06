#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "edgeinfer/preprocess.hpp"

using namespace edgeinfer;

static void BM_Preprocess_Scalar_640x480_to_224x224(benchmark::State& state) {
  const int in_w = 640, in_h = 480;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_scalar(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}

static void BM_Preprocess_Scalar_1920x1080_to_224x224(benchmark::State& state) {
  const int in_w = 1920, in_h = 1080;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_scalar(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}

#ifdef EDGEINFER_HAVE_AVX2
static void BM_Preprocess_AVX2_640x480_to_224x224(benchmark::State& state) {
  const int in_w = 640, in_h = 480;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_avx2(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}

static void BM_Preprocess_AVX2_1920x1080_to_224x224(benchmark::State& state) {
  const int in_w = 1920, in_h = 1080;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_avx2(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}
#endif

#ifdef EDGEINFER_HAVE_NEON
static void BM_Preprocess_NEON_640x480_to_224x224(benchmark::State& state) {
  const int in_w = 640, in_h = 480;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_neon(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}

static void BM_Preprocess_NEON_1920x1080_to_224x224(benchmark::State& state) {
  const int in_w = 1920, in_h = 1080;
  const int out_w = 224, out_h = 224;

  std::vector<uint8_t> image(3 * in_w * in_h);
  std::mt19937 gen(42);
  for (auto& b : image) {
    b = gen() % 256;
  }

  std::vector<float> out(3 * out_w * out_h);
  ImageView view{image.data(), in_w, in_h, 3 * in_w};
  Normalize norm;

  for (auto _ : state) {
    preprocess_neon(view, out_w, out_h, norm, out);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(3 * in_w * in_h * state.iterations());
}
#endif

BENCHMARK(BM_Preprocess_Scalar_640x480_to_224x224)->Name("BM_Preprocess/scalar/640x480");
BENCHMARK(BM_Preprocess_Scalar_1920x1080_to_224x224)->Name("BM_Preprocess/scalar/1920x1080");

#ifdef EDGEINFER_HAVE_AVX2
BENCHMARK(BM_Preprocess_AVX2_640x480_to_224x224)->Name("BM_Preprocess/avx2/640x480");
BENCHMARK(BM_Preprocess_AVX2_1920x1080_to_224x224)->Name("BM_Preprocess/avx2/1920x1080");
#endif

#ifdef EDGEINFER_HAVE_NEON
BENCHMARK(BM_Preprocess_NEON_640x480_to_224x224)->Name("BM_Preprocess/neon/640x480");
BENCHMARK(BM_Preprocess_NEON_1920x1080_to_224x224)->Name("BM_Preprocess/neon/1920x1080");
#endif
