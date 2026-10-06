add_executable(edgeinfer_bench
  bench_preprocess.cpp
  bench_queue.cpp
)

target_compile_options(edgeinfer_bench PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
target_link_libraries(edgeinfer_bench PRIVATE edgeinfer_core benchmark::benchmark_main)
