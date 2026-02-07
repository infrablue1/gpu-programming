cmake_minimum_required(VERSION 3.20)
project(gpu-programming LANGUAGES C CXX CUDA)
include_directories(./include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 110 120)

add_subdirectory(vector_add)
add_subdirectory(config)
add_subdirectory(matrix_mul)