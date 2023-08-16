cmake_minimum_required(VERSION 3.19)
project(cuda_stroke_lib LANGUAGES CUDA CXX)

option(AUTOUPDATE_STROKE_LIBRARY "Keeps whack up-to-date with origin/main, but prevents local edits. Change to OFF if you want to edit whack." ON)

include(FetchContent)
if(AUTOUPDATE_STROKE_LIBRARY)
    FetchContent_Declare(stroke
        GIT_REPOSITORY  git@github.com:cg-tuwien/cuda_stroke.git
        GIT_TAG         origin/main
        SOURCE_DIR      ${CMAKE_SOURCE_DIR}/3rdparty/stroke
        SOURCE_SUBDIR   src
    )
    FetchContent_MakeAvailable(stroke)
else()
    FetchContent_Declare(stroke
        GIT_REPOSITORY  git@github.com:cg-tuwien/cuda_stroke.git
        GIT_TAG         main
        SOURCE_DIR      ${CMAKE_SOURCE_DIR}/3rdparty/stroke
    )
    FetchContent_MakeAvailable(stroke)
endif()

add_library(differentiable_gaussian_mixture_renderer
	dgmr/forward.h	dgmr/forward.cu

)
target_link_libraries(differentiable_gaussian_mixture_renderer PRIVATE stroke PUBLIC whack)
target_include_directories(differentiable_gaussian_mixture_renderer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(differentiable_gaussian_mixture_renderer PROPERTIES CUDA_ARCHITECTURES "75;86;90")
