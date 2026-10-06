cmake_minimum_required(VERSION 3.24)
project(edge_infer_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EDGEINFER_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(EDGEINFER_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(EDGEINFER_BUILD_BENCH "Build Google Benchmark binaries" ON)
option(EDGEINFER_WITH_ORT "Build the ONNX Runtime model wrapper and server" ON)

# Warnings are errors in our own code only (dependencies are compiled with their own flags).
set(EDGEINFER_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

if(EDGEINFER_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# Which SIMD path to compile. x86-64 gets an AVX2 translation unit (selected at run time with
# __builtin_cpu_supports, so the binary still runs on CPUs without AVX2); aarch64 gets NEON,
# which is part of the base ARMv8-A ISA.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(EDGEINFER_ARCH x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(EDGEINFER_ARCH aarch64)
else()
  set(EDGEINFER_ARCH generic)
endif()
message(STATUS "edge-infer: arch=${EDGEINFER_ARCH} sanitize=${EDGEINFER_SANITIZE}")

include(cmake/Dependencies.cmake)

# ---------------------------------------------------------------- core library (no ORT)
add_library(edgeinfer_core
  src/preprocess_scalar.cpp
  src/preprocess_dispatch.cpp
  src/metrics.cpp
  src/batcher.cpp
)
if(EDGEINFER_ARCH STREQUAL "x86_64")
  target_sources(edgeinfer_core PRIVATE src/preprocess_avx2.cpp)
  set_source_files_properties(src/preprocess_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(edgeinfer_core PUBLIC EDGEINFER_HAVE_AVX2=1)
elseif(EDGEINFER_ARCH STREQUAL "aarch64")
  target_sources(edgeinfer_core PRIVATE src/preprocess_neon.cpp)
  target_compile_definitions(edgeinfer_core PUBLIC EDGEINFER_HAVE_NEON=1)
endif()
target_include_directories(edgeinfer_core PUBLIC include)
target_compile_options(edgeinfer_core PRIVATE ${EDGEINFER_WARNINGS})
find_package(Threads REQUIRED)
target_link_libraries(edgeinfer_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------- ONNX Runtime layer + server
if(EDGEINFER_WITH_ORT)
  add_library(edgeinfer_model src/model.cpp)
  target_include_directories(edgeinfer_model PUBLIC include)
  target_compile_options(edgeinfer_model PRIVATE ${EDGEINFER_WARNINGS})
  target_link_libraries(edgeinfer_model PUBLIC edgeinfer_core onnxruntime::onnxruntime nlohmann_json::nlohmann_json)

  add_executable(edge_server src/server_main.cpp)
  target_compile_options(edge_server PRIVATE ${EDGEINFER_WARNINGS})
  target_link_libraries(edge_server PRIVATE edgeinfer_model httplib::httplib)

  add_executable(edge_bench src/bench_cli.cpp)
  target_compile_options(edge_bench PRIVATE ${EDGEINFER_WARNINGS})
  target_link_libraries(edge_bench PRIVATE edgeinfer_model)

  add_executable(edge_loadgen src/loadgen.cpp)
  target_compile_options(edge_loadgen PRIVATE ${EDGEINFER_WARNINGS})
  target_link_libraries(edge_loadgen PRIVATE httplib::httplib nlohmann_json::nlohmann_json Threads::Threads)
endif()

if(EDGEINFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(EDGEINFER_BUILD_BENCH)
  add_subdirectory(bench)
endif()
