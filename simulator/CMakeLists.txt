cmake_minimum_required(VERSION 3.10)

project(mWidar_Simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Add the build configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "" FORCE)

# Right now, build for Debug 
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# When linking your target, you'll need to add:
# target_link_libraries(your_target_name ${OpenCV_LIBS})

# Add the executable
add_executable(Simulator Simulator.cpp)

# Link the libraries
target_link_libraries(Simulator ${OpenCV_LIBS})

# Add the include directories
target_include_directories(Simulator PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_include_directories(Simulator PUBLIC ${OpenCV_INCLUDE_DIRS})

# Add the compile options
target_compile_options(Simulator PUBLIC -std=c++17 -g)
