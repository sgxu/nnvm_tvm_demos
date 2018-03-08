cmake_minimum_required(VERSION 3.2)
PROJECT(tvm_pc_demo)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

option(BUILD_TEST "whether build the test exec program" ON)

add_definitions(-std=c++11)

set(TVM_HOME "/Users/xushungou/tvm")

include_directories(include)
include_directories("${TVM_HOME}/include")
include_directories("${TVM_HOME}/dmlc-core/include")
include_directories("${TVM_HOME}/dlpack/include")

file(GLOB_RECURSE cpp_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

link_directories(${TVM_HOME}/lib)
add_library(tvm_pc_demo SHARED ${cpp_SRCS} ${ocl_SRCS})
target_link_libraries(tvm_pc_demo  "-framework OpenCL" "tvm_runtime")

if(BUILD_TEST)
    add_subdirectory(./tests)
endif()
