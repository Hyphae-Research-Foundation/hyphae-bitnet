add_library(celiums-runtime-bench-impl runtime-bench.cpp)
target_link_libraries(celiums-runtime-bench-impl PRIVATE celiums-bitnet-runtime)
target_compile_features(celiums-runtime-bench-impl PRIVATE cxx_std_17)

add_executable(celiums-runtime-bench main.cpp)
target_link_libraries(celiums-runtime-bench PRIVATE celiums-runtime-bench-impl)
target_compile_features(celiums-runtime-bench PRIVATE cxx_std_17)
