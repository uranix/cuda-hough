cmake_minimum_required(VERSION 2.8)

project(Hough)

find_package(ImageMagick COMPONENTS Magick++ REQUIRED)

set(CUDA_HOST_COMPILER "/usr/bin/gcc")
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_12,code=sm_12;-Xptxas -v)

include_directories(${ImageMagick_Magick++_INCLUDE_DIR})
include_directories("/usr/include/x86_64-linux-gnu/ImageMagick-6/")
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0)

cuda_compile(HOUGH_KERNEL_O HoughKernel.cu)
add_executable(hough Hough.cpp ${HOUGH_KERNEL_O})
target_link_libraries(hough ${ImageMagick_Magick++_LIBRARY} ${CUDA_CUDART_LIBRARY})
