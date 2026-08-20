set(TARGET celiums-logits-capture)
add_executable(${TARGET} logits-capture.cpp)
target_compile_definitions(${TARGET} PRIVATE
    CELIUMS_BITNET_PRODUCT_COMMIT="${CELIUMS_BITNET_PRODUCT_COMMIT}"
    CELIUMS_BITNET_ENGINE_COMMIT="${CELIUMS_BITNET_ENGINE_COMMIT_FULL}"
    CELIUMS_BITNET_ENGINE_TREE="${CELIUMS_BITNET_ENGINE_TREE}")
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (CELIUMS_BITNET_INSTALL_COMPAT)
    install(TARGETS ${TARGET} RUNTIME)
endif()
