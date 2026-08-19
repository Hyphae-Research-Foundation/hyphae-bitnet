cmake_minimum_required(VERSION 3.22)
project(CeliumsBitNet VERSION 0.1.0 LANGUAGES C CXX)
include(CheckIncludeFileCXX)
include(CTest)

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

add_subdirectory(src)
option(CELIUMS_BITNET_BUILD_SERVER "Build the llama.cpp server" ON)
set(LLAMA_BUILD_SERVER ${CELIUMS_BITNET_BUILD_SERVER} CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "Do not install llama.cpp tests" FORCE)
add_subdirectory(3rdparty/llama.cpp)

# install

include(GNUInstallDirs)
install(FILES LICENSE LICENSE-MIT NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/celiums-bitnet)
