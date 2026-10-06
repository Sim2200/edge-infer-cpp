# All third-party code comes in through FetchContent, pinned to a release, so a clean checkout
# builds with nothing but a compiler and CMake. ONNX Runtime is the official prebuilt release
# (building it from source takes an hour); everything else is built from source.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

if(EDGEINFER_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if(EDGEINFER_BUILD_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

if(EDGEINFER_WITH_ORT)
  set(EDGEINFER_ORT_VERSION 1.20.1)
  if(EDGEINFER_ARCH STREQUAL "aarch64")
    set(_ort_pkg onnxruntime-linux-aarch64-${EDGEINFER_ORT_VERSION})
  else()
    set(_ort_pkg onnxruntime-linux-x64-${EDGEINFER_ORT_VERSION})
  endif()
  FetchContent_Declare(onnxruntime_prebuilt
    URL https://github.com/microsoft/onnxruntime/releases/download/v${EDGEINFER_ORT_VERSION}/${_ort_pkg}.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(onnxruntime_prebuilt)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_prebuilt_SOURCE_DIR}/include)

  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(nlohmann_json)

  FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(httplib)
endif()
