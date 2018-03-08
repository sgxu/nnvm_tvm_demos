
## add include path
include_directories(${PROJECT_SOURCE_DIR}/include)

## collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

## build each source file independently.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
link_directories(${TVM_HOME}/lib)

foreach(src ${srcs})
    string(REGEX MATCH "[^/]+$" src_file ${src})
    string(REPLACE ".cpp" "" basename ${src_file})
    set(exe_file_name ${PROJECT_NAME}_${basename})
    add_executable(${exe_file_name} ${src})
    target_link_libraries(${exe_file_name} tvm_pc_demo tvm_runtime)
endforeach(src)
