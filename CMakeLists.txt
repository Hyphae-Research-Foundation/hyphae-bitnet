cmake_minimum_required(VERSION 3.22)
project(CeliumsBitNetRuntime VERSION 0.2.0 LANGUAGES C CXX)
include(CheckIncludeFileCXX)
include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD
    OUTPUT_VARIABLE CELIUMS_BITNET_PRODUCT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if (NOT CELIUMS_BITNET_PRODUCT_COMMIT)
    set(CELIUMS_BITNET_PRODUCT_COMMIT "unknown")
endif()
set(CELIUMS_BITNET_ENGINE_PIN ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ENGINE_COMMIT)
if (EXISTS ${CELIUMS_BITNET_ENGINE_PIN})
    file(READ ${CELIUMS_BITNET_ENGINE_PIN} CELIUMS_BITNET_ENGINE_COMMIT_FULL)
    string(STRIP "${CELIUMS_BITNET_ENGINE_COMMIT_FULL}" CELIUMS_BITNET_ENGINE_COMMIT_FULL)
    string(SUBSTRING "${CELIUMS_BITNET_ENGINE_COMMIT_FULL}" 0 9 CELIUMS_BITNET_ENGINE_COMMIT)
else()
    execute_process(
        COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp rev-parse --short=9 HEAD
        OUTPUT_VARIABLE CELIUMS_BITNET_ENGINE_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
if (NOT CELIUMS_BITNET_ENGINE_COMMIT)
    set(CELIUMS_BITNET_ENGINE_COMMIT "unknown")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# option list
option(BITNET_ARM_TL1    "Celiums BitNet: use TL1 on ARM"          OFF)
option(BITNET_X86_TL2    "Celiums BitNet: use TL2 on x86"          OFF)
option(CELIUMS_BITNET_STRICT "Preserve the Celiums BitNet numerical contract" ON)
option(CELIUMS_BITNET_EXPERIMENTAL "Build experimental kernels" OFF)
option(CELIUMS_BITNET_BUILD_SERVER "Build the Celiums BitNet server" ON)
option(CELIUMS_BITNET_INSTALL_COMPAT "Install llama.cpp compatibility tools and development files" OFF)
set(CELIUMS_BITNET_CPU_PROFILE "native" CACHE STRING "CPU profile: native, avx2, or scalar")
set_property(CACHE CELIUMS_BITNET_CPU_PROFILE PROPERTY STRINGS native avx2 scalar)

if (NOT CELIUMS_BITNET_STRICT)
    message(FATAL_ERROR "Non-strict builds are not implemented; keep CELIUMS_BITNET_STRICT=ON")
endif()

if ((BITNET_ARM_TL1 OR BITNET_X86_TL2) AND NOT CELIUMS_BITNET_EXPERIMENTAL)
    message(FATAL_ERROR "TL1/TL2 require CELIUMS_BITNET_EXPERIMENTAL=ON")
endif()

if (NOT CELIUMS_BITNET_CPU_PROFILE MATCHES "^(native|avx2|scalar)$")
    message(FATAL_ERROR "CELIUMS_BITNET_CPU_PROFILE must be native, avx2, or scalar")
endif()

if (CELIUMS_BITNET_CPU_PROFILE STREQUAL "native")
    set(GGML_NATIVE ON CACHE BOOL "Use the native CPU ISA" FORCE)
elseif (CELIUMS_BITNET_CPU_PROFILE STREQUAL "avx2")
    set(GGML_NATIVE OFF CACHE BOOL "Use the native CPU ISA" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "Enable SSE 4.2" FORCE)
    set(GGML_AVX ON CACHE BOOL "Enable AVX" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "Enable AVX2" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "Disable AVX-VNNI" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "Disable AVX-512" FORCE)
    set(GGML_FMA ON CACHE BOOL "Enable FMA" FORCE)
    set(GGML_F16C ON CACHE BOOL "Enable F16C" FORCE)
elseif (CELIUMS_BITNET_CPU_PROFILE STREQUAL "scalar")
    set(GGML_NATIVE OFF CACHE BOOL "Use the native CPU ISA" FORCE)
    set(GGML_SSE42 OFF CACHE BOOL "Disable SSE 4.2" FORCE)
    set(GGML_AVX OFF CACHE BOOL "Disable AVX" FORCE)
    set(GGML_AVX2 OFF CACHE BOOL "Disable AVX2" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "Disable AVX-VNNI" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "Disable AVX-512" FORCE)
    set(GGML_FMA OFF CACHE BOOL "Disable FMA" FORCE)
    set(GGML_F16C OFF CACHE BOOL "Disable F16C" FORCE)
    set(GGML_BMI2 OFF CACHE BOOL "Disable BMI2" FORCE)
endif()


set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# override ggml options
set(GGML_BITNET_ARM_TL1    ${BITNET_ARM_TL1})
set(GGML_BITNET_X86_TL2    ${BITNET_X86_TL2})

if (GGML_BITNET_ARM_TL1)
    add_compile_definitions(GGML_BITNET_ARM_TL1)
endif()
if (GGML_BITNET_X86_TL2)
    add_compile_definitions(GGML_BITNET_X86_TL2)
endif()
if (CELIUMS_BITNET_STRICT)
    add_compile_definitions(CELIUMS_BITNET_STRICT)
endif()
if (CELIUMS_BITNET_EXPERIMENTAL)
    add_compile_definitions(CELIUMS_BITNET_EXPERIMENTAL)
endif()

find_package(Threads REQUIRED)

# Inherited llama.cpp server and CLI targets are compatibility-only.
set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build internal llama.cpp server components" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "Do not install llama.cpp tests" FORCE)
set(LLAMA_TOOLS_INSTALL ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Install llama.cpp compatibility tools" FORCE)
set(CELIUMS_BITNET_ENGINE_INSTALL ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Install llama.cpp engine development files" FORCE)
add_subdirectory(3rdparty/llama.cpp)
add_subdirectory(src)
if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if (NOT TARGET llama-bench-impl)
    message(FATAL_ERROR "Celiums BitNet Runtime requires LLAMA_BUILD_COMMON=ON and LLAMA_BUILD_TOOLS=ON")
endif()

add_dependencies(celiums-runtime-bench celiums-bitnet-runtime)

add_executable(celiums-bitnet app/celiums-bitnet.cpp)
target_compile_features(celiums-bitnet PRIVATE cxx_std_17)
target_compile_definitions(celiums-bitnet PRIVATE
    CELIUMS_BITNET_RUNTIME_VERSION="${PROJECT_VERSION}"
    CELIUMS_BITNET_RUNTIME_PROFILE="${CELIUMS_BITNET_CPU_PROFILE}"
    CELIUMS_BITNET_PRODUCT_COMMIT="${CELIUMS_BITNET_PRODUCT_COMMIT}"
    CELIUMS_BITNET_RUNTIME_STRICT=1)
target_link_libraries(celiums-bitnet PRIVATE
    celiums-bitnet-runtime
    celiums-runtime-bench-impl)
if (CELIUMS_BITNET_BUILD_SERVER)
    target_compile_definitions(celiums-bitnet PRIVATE CELIUMS_BITNET_RUNTIME_SERVER=1)
    target_link_libraries(celiums-bitnet PRIVATE celiums-runtime-server-impl)
endif()

set_target_properties(celiums-bitnet PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime")

install(TARGETS celiums-bitnet RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS celiums-bitnet-runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/celiums)

set(CELIUMS_BITNET_PRIVATE_TARGETS
    celiums-runtime-bench-impl llama llama-common
    ggml ggml-base ggml-cpu)
if (CELIUMS_BITNET_BUILD_SERVER)
    list(APPEND CELIUMS_BITNET_PRIVATE_TARGETS celiums-runtime-server-impl)
endif()
if (NOT CELIUMS_BITNET_INSTALL_COMPAT)
    set_target_properties(llama ggml PROPERTIES PUBLIC_HEADER "")
    if (TARGET mtmd)
        set_target_properties(mtmd PROPERTIES PUBLIC_HEADER "" PRIVATE_HEADER "")
    endif()
endif()
foreach(target IN LISTS CELIUMS_BITNET_PRIVATE_TARGETS)
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN")
    get_target_property(target_type ${target} TYPE)
    if (target_type STREQUAL "STATIC_LIBRARY")
        continue()
    endif()
    install(TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

set_target_properties(celiums-bitnet-runtime PROPERTIES INSTALL_RPATH "$ORIGIN")

configure_package_config_file(
    cmake/CeliumsBitNetRuntimeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CeliumsBitNetRuntimeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeliumsBitNetRuntime)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CeliumsBitNetRuntimeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CeliumsBitNetRuntimeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CeliumsBitNetRuntimeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeliumsBitNetRuntime)

# install

install(FILES LICENSE LICENSE-MIT LICENSE-BSD-3-Clause NOTICE
    3rdparty/llama.cpp/NOTICE-CELIUMS
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime)
install(FILES 3rdparty/llama.cpp/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime
    RENAME LICENSE-LLAMA-MIT)
install(FILES 3rdparty/llama.cpp/vendor/cpp-httplib/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime
    RENAME LICENSE-CPP-HTTPLIB)
install(FILES 3rdparty/llama.cpp/licenses/LICENSE-jsonhpp
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime)

if (BUILD_TESTING)
    add_executable(test-celiums-runtime-api tests/test-runtime-api.c)
    target_link_libraries(test-celiums-runtime-api PRIVATE celiums-bitnet-runtime)
    add_test(NAME test-celiums-runtime-api COMMAND test-celiums-runtime-api)
    add_executable(test-celiums-runtime-session tests/test-runtime-session.cpp)
    target_link_libraries(test-celiums-runtime-session PRIVATE celiums-bitnet-runtime)
    if (EXISTS ${PROJECT_SOURCE_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)
        add_test(NAME test-celiums-runtime-session COMMAND test-celiums-runtime-session
            ${PROJECT_SOURCE_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)
    endif()
endif()
