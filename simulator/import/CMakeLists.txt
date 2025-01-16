# Use Eigen Library and compile import_bin 
cmake_minimum_required(VERSION 3.10)

project(Importing_Bin_Files)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test import_bin.cpp)

target_include_directories(test PUBLIC ${EIGEN3_INCLUDE_DIRS})

target_compile_options(test PUBLIC --std=c++17 -g)

