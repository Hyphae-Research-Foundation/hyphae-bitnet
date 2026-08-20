set(TARGET celiums-logits-capture)
add_executable(${TARGET} logits-capture.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (CELIUMS_BITNET_INSTALL_COMPAT)
    install(TARGETS ${TARGET} RUNTIME)
endif()
