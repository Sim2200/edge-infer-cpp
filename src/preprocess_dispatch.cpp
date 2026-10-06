// Run-time selection of the preprocessing path. EDGEINFER_PREPROCESS=scalar forces the reference
// (used by the benchmarks and to rule the kernel out when debugging a model mismatch).
#include <cstdlib>
#include <cstring>

#include "edgeinfer/preprocess.hpp"

namespace edgeinfer {

namespace {
using Fn = void (*)(const ImageView&, int, int, const Normalize&, std::span<float>);

struct Choice {
  Fn fn;
  std::string_view name;
};

Choice choose() {
  const char* force = std::getenv("EDGEINFER_PREPROCESS");
  const bool scalar_only = force != nullptr && std::strcmp(force, "scalar") == 0;
  if (!scalar_only) {
#if defined(EDGEINFER_HAVE_AVX2)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return {&preprocess_avx2, "avx2"};
#endif
#if defined(EDGEINFER_HAVE_NEON)
    return {&preprocess_neon, "neon"};
#endif
  }
  return {&preprocess_scalar, "scalar"};
}

const Choice& choice() {
  static const Choice c = choose();  // thread-safe initialisation (C++11 magic statics)
  return c;
}
}  // namespace

void preprocess(const ImageView& in, int out_w, int out_h, const Normalize& norm, std::span<float> out) {
  choice().fn(in, out_w, out_h, norm, out);
}

std::string_view selected_path() { return choice().name; }

}  // namespace edgeinfer
