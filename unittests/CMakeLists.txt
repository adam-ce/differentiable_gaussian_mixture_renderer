cmake_minimum_required(VERSION 3.18)
project(dgmr_unittests LANGUAGES CUDA CXX)

FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
)
FetchContent_MakeAvailable(catch2)
find_package(CUDAToolkit)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;90")
endif()

add_executable(dgmr_unittests
    math.cpp
    raster_bin_sizer.cpp
    grad_math.cu
    unit_test_utils.h
)

target_link_libraries(dgmr_unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt differentiable_gaussian_mixture_renderer)
