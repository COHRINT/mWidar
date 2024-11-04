cmake_minimum_required(VERSION 3.29)
project(mWidar)

include(FetchContent)

# Define Eigen as a dependency and download it if necessary
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find TBB
find_package(TBB REQUIRED)

find_package(OpenCV REQUIRED)

# Fetch zlib
FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        master  # You can specify a specific version or tag here
)
FetchContent_MakeAvailable(zlib)

# Define Matio as a dependency
FetchContent_Declare(
        matio
        GIT_REPOSITORY https://github.com/tbeu/matio.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(matio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Specify source files
set(SOURCES

        src/test-scripts/testBasicConsumer.cpp
        src/GraphProcessor.cpp
        src/Object.cpp
        src/VisionProcessor.cpp
        src/DataProcessor.cpp
        include/GraphProcessor.h
)

set(HEADERS
        include/GraphProcessor.h
        include/Object.h
        include/visionProcessor.h
        include/DataProcessor.h
        include/json.hpp
)

# Create executable
add_executable(mWidar ${SOURCES})

# Include directories for libraries
target_include_directories(mWidar PRIVATE
        ${eigen_SOURCE_DIR}
        ${opencv_SOURCE_DIR}/include
        ${matio_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(mWidar PRIVATE
        ${OpenCV_LIBS}
        matio Eigen3::Eigen
)


