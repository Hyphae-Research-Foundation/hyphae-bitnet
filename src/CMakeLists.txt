if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Celiums BitNet")
endif()

add_library(celiums-bitnet-runtime
    celiums-bitnet-runtime.cpp
    ${PROJECT_SOURCE_DIR}/include/celiums/bitnet_runtime.h)
add_library(Celiums::BitNetRuntime ALIAS celiums-bitnet-runtime)

set_target_properties(celiums-bitnet-runtime PROPERTIES
    OUTPUT_NAME celiums-bitnet-runtime
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/include/celiums/bitnet_runtime.h
    BUILD_RPATH "$ORIGIN")

target_compile_features(celiums-bitnet-runtime PRIVATE cxx_std_17)
target_compile_definitions(celiums-bitnet-runtime PRIVATE
    CELIUMS_BITNET_RUNTIME_BUILD
    CELIUMS_BITNET_RUNTIME_VERSION="${PROJECT_VERSION}"
    CELIUMS_BITNET_RUNTIME_PROFILE="${CELIUMS_BITNET_CPU_PROFILE}"
    CELIUMS_BITNET_PRODUCT_COMMIT="${CELIUMS_BITNET_PRODUCT_COMMIT}"
    CELIUMS_BITNET_ENGINE_COMMIT="${CELIUMS_BITNET_ENGINE_COMMIT}")
target_include_directories(celiums-bitnet-runtime
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(celiums-bitnet-runtime PRIVATE llama llama-common)
