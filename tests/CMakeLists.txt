include(GoogleTest)

add_executable(edgeinfer_tests
  bounded_queue_test.cpp
  mpmc_ring_test.cpp
  thread_pool_test.cpp
  preprocess_test.cpp
  batcher_test.cpp
  metrics_test.cpp
)

target_compile_options(edgeinfer_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
target_link_libraries(edgeinfer_tests PRIVATE edgeinfer_core GTest::gtest_main)
target_compile_definitions(edgeinfer_tests PRIVATE "EDGEINFER_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

gtest_discover_tests(edgeinfer_tests DISCOVERY_TIMEOUT 60)

if(TARGET edgeinfer_model)
  add_executable(edgeinfer_model_tests
    model_test.cpp
  )
  target_compile_options(edgeinfer_model_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  target_link_libraries(edgeinfer_model_tests PRIVATE edgeinfer_model GTest::gtest_main)
  target_compile_definitions(edgeinfer_model_tests PRIVATE "EDGEINFER_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")

  gtest_discover_tests(edgeinfer_model_tests DISCOVERY_TIMEOUT 60)
endif()
