# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project(ASS4)

find_package(CUDA QUIET REQUIRED)

set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)


set(CUDA_SOURCE
        ${SRC_DIR}/main.cu
        ${SRC_DIR}/image.cpp
        ${SRC_DIR}/conv_separable_host.cpp
        )

IF (WIN32)
ELSE()
    set( CMAKE_CXX_FLAGS "-g -W -Wall -std=c++11 -DDEBUG -Wno-unused-variable")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set( CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets --expt-relaxed-constexpr --resource-usage -arch=sm_53 -lineinfo")

cuda_add_executable( ASS4 ${CUDA_SOURCE})

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
