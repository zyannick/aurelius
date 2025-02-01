cmake_minimum_required(VERSION 3.10)
include(ExternalProject)
project(Aurelius)

find_package(Eigen3 REQUIRED) 

find_package(Git REQUIRED)

ExternalProject_Add(libsleef
    GIT_REPOSITORY https://github.com/shibatch/sleef
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/contrib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")

include_directories(${CMAKE_BINARY_DIR}/contrib/include)
link_directories(${CMAKE_BINARY_DIR}/contrib/lib)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MyProject src/main.cpp)