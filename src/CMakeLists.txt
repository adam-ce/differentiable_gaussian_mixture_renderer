cmake_minimum_required(VERSION 3.19)
project(cuda_stroke_lib LANGUAGES CUDA CXX)

option(AUTOUPDATE_STROKE_LIBRARY "Keeps whack up-to-date with origin/main, but prevents local edits. Change to OFF if you want to edit whack." ON)
set(A_PYTHON_ENV_PATH "$ENV{HOME}/.local/" CACHE PATH "Path to your python environment (pip, conda or similar)")
set(A_PYTHON_ENV_SITE_PACKAGES_PATH "$ENV{HOME}/.local/lib/python3.10/site-packages/" CACHE PATH "Path to your python site-packages directory (python version dependent, try echo $PYTHONPATH on the terminal)")

list(APPEND CMAKE_PREFIX_PATH "${A_PYTHON_ENV_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${A_PYTHON_ENV_SITE_PACKAGES_PATH}/torch")


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;90")
endif()

find_package(Torch REQUIRED)

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
	dgmr/splat_forward.h	dgmr/splat_forward.cu
	dgmr/constants.h

)
target_link_libraries(differentiable_gaussian_mixture_renderer PUBLIC stroke whack)
target_include_directories(differentiable_gaussian_mixture_renderer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
#set_target_properties(differentiable_gaussian_mixture_renderer PROPERTIES CUDA_ARCHITECTURES )
