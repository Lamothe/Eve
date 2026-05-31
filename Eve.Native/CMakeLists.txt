cmake_minimum_required(VERSION 3.14)
project(eve_native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to llama.cpp's built GGML
set(GGML_DIR "/home/michael/Projects/llama.cpp")
set(GGML_BUILD_DIR "${GGML_DIR}/build")
set(GGML_INCLUDE_DIR "${GGML_DIR}/ggml/include")

# GGML library paths
set(GGML_BASE_LIB "${GGML_BUILD_DIR}/bin/libggml-base.so")
set(GGML_CPU_LIB  "${GGML_BUILD_DIR}/bin/libggml-cpu.so")
set(GGML_VULKAN_LIB "${GGML_BUILD_DIR}/bin/libggml-vulkan.so")

# Our include
include_directories(${GGML_INCLUDE_DIR})
include_directories(include)

# Shared library target
add_library(eve_native SHARED
    src/eve_native.cpp
)

# Link GGML libraries
target_link_libraries(eve_native PRIVATE
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
    ${GGML_VULKAN_LIB}
    m
    pthread
    dl
)

# Copy built library to Eve project
set_target_properties(eve_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../lib"
)

install(TARGETS eve_native
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/../lib"
)
