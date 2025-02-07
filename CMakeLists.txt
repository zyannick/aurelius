cmake_minimum_required(VERSION 3.10)
project(Aurelius)

find_package(Eigen3 REQUIRED) 
find_package(SLEEF REQUIRED) 

find_package(Git REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MyProject src/main.cpp)