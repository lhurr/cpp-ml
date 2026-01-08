cmake_minimum_required(VERSION 3.10)
project(SimpleNeuralNet)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(xor_demo examples/xor_main.cpp)
