cmake_minimum_required(VERSION 3.5.0)
project(cuHECNN VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
endif()

include_directories(src)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
find_package(CUDAToolkit)

add_executable(cuHECNN main.cu src/hmm.cu src/inferNew.cu)

target_link_libraries(cuHECNN libtroy.so)