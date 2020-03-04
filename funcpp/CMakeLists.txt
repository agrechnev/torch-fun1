cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(torch_cppfun)

find_package(Torch REQUIRED)

add_executable(fun fun.cpp)
target_link_libraries(fun ${TORCH_LIBRARIES})
