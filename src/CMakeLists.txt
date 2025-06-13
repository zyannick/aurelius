cmake_minimum_required(VERSION 3.10)
project(Aurelius)

file(GLOB_RECURSE AURELIUS_SOURCES
    *.cc
    modules/*.cc
    examples/*.cc
)

list(FILTER AURELIUS_SOURCES EXCLUDE REGEX ".*/third_party/.*")

add_executable(Aurelius ${AURELIUS_SOURCES})

include_directories($ENV{HOME}/.local/include/)

find_package(Eigen3 REQUIRED) 
find_package(Git REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(aurelius main.cc)