add_library(celiums-runtime-server-impl runtime-server.cpp)
target_include_directories(celiums-runtime-server-impl PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/llama.cpp/vendor
    ${PROJECT_SOURCE_DIR}/3rdparty/llama.cpp/vendor/cpp-httplib)
target_link_libraries(celiums-runtime-server-impl PRIVATE
    celiums-bitnet-runtime
    cpp-httplib
    Threads::Threads)
target_compile_features(celiums-runtime-server-impl PRIVATE cxx_std_17)

add_executable(celiums-runtime-server main.cpp)
target_link_libraries(celiums-runtime-server PRIVATE celiums-runtime-server-impl)
target_compile_features(celiums-runtime-server PRIVATE cxx_std_17)
