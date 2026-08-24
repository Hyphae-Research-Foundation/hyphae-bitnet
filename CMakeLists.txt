cmake_minimum_required(VERSION 3.22)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" CELIUMS_BITNET_VERSION)
string(STRIP "${CELIUMS_BITNET_VERSION}" CELIUMS_BITNET_VERSION)
if (NOT CELIUMS_BITNET_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must contain a semantic version")
endif()
project(HyphaeBitNetRuntime VERSION ${CELIUMS_BITNET_VERSION} LANGUAGES C CXX)
include(CheckIncludeFileCXX)
include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD
    OUTPUT_VARIABLE CELIUMS_BITNET_PRODUCT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if (NOT CELIUMS_BITNET_PRODUCT_COMMIT)
    set(CELIUMS_BITNET_PRODUCT_COMMIT "unknown")
endif()
set(CELIUMS_BITNET_ENGINE_PIN "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ENGINE_COMMIT")
set(CELIUMS_BITNET_ENGINE_TREE_PIN "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ENGINE_TREE")
if (NOT EXISTS "${CELIUMS_BITNET_ENGINE_PIN}" OR NOT EXISTS "${CELIUMS_BITNET_ENGINE_TREE_PIN}")
    message(FATAL_ERROR "Vendored engine provenance files are required")
endif()
file(READ "${CELIUMS_BITNET_ENGINE_PIN}" CELIUMS_BITNET_ENGINE_COMMIT_FULL)
file(READ "${CELIUMS_BITNET_ENGINE_TREE_PIN}" CELIUMS_BITNET_ENGINE_TREE)
string(STRIP "${CELIUMS_BITNET_ENGINE_COMMIT_FULL}" CELIUMS_BITNET_ENGINE_COMMIT_FULL)
string(STRIP "${CELIUMS_BITNET_ENGINE_TREE}" CELIUMS_BITNET_ENGINE_TREE)
string(LENGTH "${CELIUMS_BITNET_ENGINE_COMMIT_FULL}" CELIUMS_BITNET_ENGINE_COMMIT_LENGTH)
string(LENGTH "${CELIUMS_BITNET_ENGINE_TREE}" CELIUMS_BITNET_ENGINE_TREE_LENGTH)
if (NOT CELIUMS_BITNET_ENGINE_COMMIT_LENGTH EQUAL 40 OR
        NOT CELIUMS_BITNET_ENGINE_COMMIT_FULL MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "3rdparty/llama.cpp/ENGINE_COMMIT must contain a full Git commit")
endif()
if (NOT CELIUMS_BITNET_ENGINE_TREE_LENGTH EQUAL 40 OR
        NOT CELIUMS_BITNET_ENGINE_TREE MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "cmake/ENGINE_TREE must contain a full Git tree hash")
endif()
string(SUBSTRING "${CELIUMS_BITNET_ENGINE_COMMIT_FULL}" 0 9 CELIUMS_BITNET_ENGINE_COMMIT)

execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --is-inside-work-tree
    OUTPUT_VARIABLE CELIUMS_BITNET_GIT_WORK_TREE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --show-toplevel
    OUTPUT_VARIABLE CELIUMS_BITNET_GIT_TOPLEVEL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if (CELIUMS_BITNET_VERIFY_ENGINE_TREE AND CELIUMS_BITNET_GIT_WORK_TREE STREQUAL "true" AND
        CELIUMS_BITNET_GIT_TOPLEVEL STREQUAL CMAKE_CURRENT_SOURCE_DIR AND
        EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compute-engine-tree.sh")
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compute-engine-tree.sh"
        OUTPUT_VARIABLE CELIUMS_BITNET_ENGINE_TREE_ACTUAL
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CELIUMS_BITNET_ENGINE_TREE_RESULT
        ERROR_QUIET)
    if (NOT CELIUMS_BITNET_ENGINE_TREE_RESULT EQUAL 0)
        message(FATAL_ERROR "Unable to validate the vendored engine tree")
    endif()
    if (NOT CELIUMS_BITNET_ENGINE_TREE_ACTUAL STREQUAL CELIUMS_BITNET_ENGINE_TREE)
        message(FATAL_ERROR "Vendored engine tree does not match cmake/ENGINE_TREE")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT BUILD_SHARED_LIBS)
    message(STATUS "Static builds are internal-only; public SDK install rules are disabled")
endif()

# option list
option(BITNET_ARM_TL1    "Celiums BitNet: use TL1 on ARM"          OFF)
option(BITNET_X86_TL2    "Celiums BitNet: use TL2 on x86"          OFF)
option(CELIUMS_BITNET_STRICT "Preserve the Celiums BitNet numerical contract" ON)
option(CELIUMS_BITNET_EXPERIMENTAL "Build experimental kernels" OFF)
option(CELIUMS_BITNET_BUILD_SERVER "Build the Celiums BitNet server" ON)
option(CELIUMS_BITNET_BUILD_GATEWAY "Build the optional Rust Hyphae gateway" OFF)
option(CELIUMS_BITNET_INSTALL_COMPAT "Install llama.cpp compatibility tools and development files" OFF)
option(CELIUMS_BITNET_VERIFY_ENGINE_TREE "Verify worktree contents against cmake/ENGINE_TREE" OFF)
set(CELIUMS_BITNET_TEST_MODEL "" CACHE FILEPATH "Strict I2_S model used by integration tests")
set(CELIUMS_BONSAI_TEST_MODEL "" CACHE FILEPATH "Bonsai Qwen35 Q1_0 model used by integration tests")

if (NOT BUILD_SHARED_LIBS AND CELIUMS_BITNET_INSTALL_COMPAT)
    message(FATAL_ERROR "Static builds cannot install the public or compatibility SDK")
endif()
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
    if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        message(FATAL_ERROR "The avx2 CPU profile requires x86_64")
    endif()
    set(GGML_NATIVE OFF CACHE BOOL "Use the native CPU ISA" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "Enable SSE 4.2" FORCE)
    set(GGML_AVX ON CACHE BOOL "Enable AVX" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "Enable AVX2" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "Disable AVX-VNNI" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "Disable AVX-512" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "Disable AVX-512 VBMI" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "Disable AVX-512 VNNI" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "Disable AVX-512 BF16" FORCE)
    set(GGML_AMX_TILE OFF CACHE BOOL "Disable AMX tile" FORCE)
    set(GGML_AMX_INT8 OFF CACHE BOOL "Disable AMX INT8" FORCE)
    set(GGML_AMX_BF16 OFF CACHE BOOL "Disable AMX BF16" FORCE)
    set(GGML_BMI2 OFF CACHE BOOL "Disable BMI2" FORCE)
    set(GGML_FMA ON CACHE BOOL "Enable FMA" FORCE)
    set(GGML_F16C ON CACHE BOOL "Enable F16C" FORCE)
elseif (CELIUMS_BITNET_CPU_PROFILE STREQUAL "scalar")
    set(GGML_NATIVE OFF CACHE BOOL "Use the native CPU ISA" FORCE)
    set(GGML_SSE42 OFF CACHE BOOL "Disable SSE 4.2" FORCE)
    set(GGML_AVX OFF CACHE BOOL "Disable AVX" FORCE)
    set(GGML_AVX2 OFF CACHE BOOL "Disable AVX2" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "Disable AVX-VNNI" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "Disable AVX-512" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "Disable AVX-512 VBMI" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "Disable AVX-512 VNNI" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "Disable AVX-512 BF16" FORCE)
    set(GGML_AMX_TILE OFF CACHE BOOL "Disable AMX tile" FORCE)
    set(GGML_AMX_INT8 OFF CACHE BOOL "Disable AMX INT8" FORCE)
    set(GGML_AMX_BF16 OFF CACHE BOOL "Disable AMX BF16" FORCE)
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

set(GGML_BLAS OFF CACHE BOOL "Disable unsupported BLAS backend" FORCE)
set(GGML_CUDA OFF CACHE BOOL "Disable unsupported CUDA backend" FORCE)
set(GGML_HIP OFF CACHE BOOL "Disable unsupported HIP backend" FORCE)
set(GGML_METAL OFF CACHE BOOL "Disable unsupported Metal backend" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Disable unsupported Vulkan backend" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "Disable unsupported OpenCL backend" FORCE)
set(GGML_SYCL OFF CACHE BOOL "Disable unsupported SYCL backend" FORCE)
set(GGML_CANN OFF CACHE BOOL "Disable unsupported CANN backend" FORCE)
set(GGML_RPC OFF CACHE BOOL "Disable unsupported RPC backend" FORCE)
set(GGML_OPENVINO OFF CACHE BOOL "Disable unsupported OpenVINO backend" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build internal llama.cpp common library" FORCE)
set(LLAMA_BUILD_TOOLS ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Build llama.cpp compatibility tools" FORCE)
set(LLAMA_BUILD_SERVER ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Build llama.cpp compatibility server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build the llama.cpp unified application" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "Do not build the llama.cpp web UI" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "Do not download the llama.cpp web UI" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "Disable HTTPS in the private HTTP dependency" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "Do not install llama.cpp tests" FORCE)
set(LLAMA_TOOLS_INSTALL ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Install llama.cpp compatibility tools" FORCE)
set(CELIUMS_BITNET_ENGINE_INSTALL ${CELIUMS_BITNET_INSTALL_COMPAT} CACHE BOOL "Install llama.cpp engine development files" FORCE)
set(LLAMA_BUILD_COMMIT ${CELIUMS_BITNET_ENGINE_COMMIT_FULL})
set(LLAMA_BUILD_NUMBER 0)
add_subdirectory(3rdparty/llama.cpp)
add_subdirectory(src)
add_subdirectory(tools)

if (CELIUMS_BITNET_BUILD_GATEWAY)
    find_program(CELIUMS_CARGO_EXECUTABLE cargo REQUIRED)
    set(CELIUMS_GATEWAY_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway/Cargo.toml")
    set(CELIUMS_GATEWAY_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-gateway")
    file(GLOB_RECURSE CELIUMS_GATEWAY_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway/src/*.rs")
    set(CELIUMS_GATEWAY_BINARIES
        "${CELIUMS_GATEWAY_TARGET_DIR}/release/celiums-runtime-gateway"
        "${CELIUMS_GATEWAY_TARGET_DIR}/release/celiums-hyphae-sidecar"
        "${CELIUMS_GATEWAY_TARGET_DIR}/release/celiums-runtime-mcp")
    add_custom_command(
        OUTPUT ${CELIUMS_GATEWAY_BINARIES}
        COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${CELIUMS_GATEWAY_TARGET_DIR}
            CARGO_INCREMENTAL=0
            ${CELIUMS_CARGO_EXECUTABLE} build --manifest-path ${CELIUMS_GATEWAY_MANIFEST}
                --release --locked --bins
        DEPENDS
            ${CELIUMS_GATEWAY_SOURCES}
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway/Cargo.toml"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway/Cargo.lock"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway/rust-toolchain.toml"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools/runtime-gateway"
        COMMENT "Building Celiums Runtime Gateway")
    add_custom_target(celiums-runtime-gateway-binaries ALL DEPENDS ${CELIUMS_GATEWAY_BINARIES})
    if (BUILD_SHARED_LIBS)
        install(PROGRAMS ${CELIUMS_GATEWAY_BINARIES} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
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
    OUTPUT_NAME hyphae-bitnet
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime")

if (BUILD_SHARED_LIBS)
install(TARGETS celiums-bitnet RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(CODE "file(CREATE_LINK hyphae-bitnet \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/celiums-bitnet\" SYMBOLIC)")
install(TARGETS celiums-bitnet-runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/celiums)

add_library(celiums-bitnet-sdk INTERFACE)
set_target_properties(celiums-bitnet-sdk PROPERTIES EXPORT_NAME BitNetRuntime)
target_include_directories(celiums-bitnet-sdk INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(celiums-bitnet-sdk INTERFACE
    $<BUILD_INTERFACE:celiums-bitnet-runtime>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/celiums-bitnet-runtime/${CMAKE_SHARED_LIBRARY_PREFIX}celiums-bitnet-runtime${CMAKE_SHARED_LIBRARY_SUFFIX}>)
install(TARGETS celiums-bitnet-sdk EXPORT CeliumsBitNetRuntimeTargets)

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
install(EXPORT CeliumsBitNetRuntimeTargets
    FILE CeliumsBitNetRuntimeTargets.cmake
    NAMESPACE Celiums::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeliumsBitNetRuntime)

# install

install(FILES LICENSE LICENSE-MIT LICENSE-BSD-3-Clause NOTICE
    3rdparty/llama.cpp/NOTICE-CELIUMS
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime)
if (CELIUMS_BITNET_BUILD_GATEWAY)
    install(FILES LICENSE-HYPHAE-APACHE-2.0 NOTICE-HYPHAE NOTICE-RUST-DEPENDENCIES
        tools/runtime-gateway/Cargo.lock
        tools/runtime-gateway/THIRD_PARTY_LICENSES.html
        DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime)
endif()
install(FILES 3rdparty/llama.cpp/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime
    RENAME LICENSE-LLAMA-MIT)
install(FILES 3rdparty/llama.cpp/vendor/cpp-httplib/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime
    RENAME LICENSE-CPP-HTTPLIB)
install(FILES 3rdparty/llama.cpp/licenses/LICENSE-jsonhpp
    DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet-runtime)
endif()

if (BUILD_TESTING)
    add_executable(test-celiums-runtime-api tests/test-runtime-api.c)
    target_compile_definitions(test-celiums-runtime-api PRIVATE
        CELIUMS_BITNET_TEST_VERSION="${PROJECT_VERSION}")
    target_link_libraries(test-celiums-runtime-api PRIVATE celiums-bitnet-runtime)
    add_test(NAME test-celiums-runtime-api COMMAND test-celiums-runtime-api)
    add_executable(test-celiums-runtime-session tests/test-runtime-session.cpp)
    target_link_libraries(test-celiums-runtime-session PRIVATE celiums-bitnet-runtime)
    if (NOT CELIUMS_BITNET_TEST_MODEL)
        set(CELIUMS_BITNET_TEST_MODEL ${PROJECT_SOURCE_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf)
    endif()
    if (EXISTS ${CELIUMS_BITNET_TEST_MODEL})
        add_test(NAME test-celiums-runtime-session COMMAND test-celiums-runtime-session
            ${CELIUMS_BITNET_TEST_MODEL} bitnet)
    endif()
    if (CELIUMS_BONSAI_TEST_MODEL AND EXISTS ${CELIUMS_BONSAI_TEST_MODEL})
        add_test(NAME test-celiums-runtime-session-bonsai COMMAND test-celiums-runtime-session
            ${CELIUMS_BONSAI_TEST_MODEL} bonsai)
    endif()
    add_executable(test-q1-repack-oracle tests/test-q1-repack-oracle.cpp)
    target_include_directories(test-q1-repack-oracle PRIVATE
        ${PROJECT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
        ${PROJECT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src
        ${PROJECT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src/ggml-cpu)
    target_link_libraries(test-q1-repack-oracle PRIVATE ggml-cpu ggml-base celiums-exact)
    add_test(NAME test-q1-repack-oracle COMMAND test-q1-repack-oracle)
endif()
