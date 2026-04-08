cmake_minimum_required(VERSION 3.18)
project(yolo_engine_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# --- Dependencies ---
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)  # pip install pybind11

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include /usr/include/aarch64-linux-gnu)
find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    HINTS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found")
endif()

# Jetson ORIN SM 8.7 + Xavier SM 7.2
set(CMAKE_CUDA_ARCHITECTURES 72 87)

# --- Core C++ engine (static lib) ---
add_library(yolo_engine STATIC
    src/preprocess.cu
    src/postprocess.cpp
    src/yolo_engine.cpp
)
target_include_directories(yolo_engine PUBLIC
    src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_engine PUBLIC
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
)

# --- Python module ---
pybind11_add_module(yolo_engine_cpp src/pybind_wrapper.cpp)
target_link_libraries(yolo_engine_cpp PRIVATE yolo_engine)
