cmake_minimum_required(VERSION 3.18)
project(differentiable_gaussian_mixture_renderer LANGUAGES CUDA CXX)

option(BUILD_DGMR_UNITTESTS "include unit test targets for stroke in the buildsystem" OFF)

add_subdirectory(src)
if(BUILD_DGMR_UNITTESTS OR BUILD_TESTING)
    add_subdirectory(unittests)
endif()
