cmake_minimum_required(VERSION 3.10)
project(l1MedialSkeleton)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

add_library(l1MedialSkeleton MODULE l1MedialSkeleton.cpp)

# Ensure no "lib" prefix is added
set_target_properties(l1MedialSkeleton PROPERTIES PREFIX "")

target_link_libraries(l1MedialSkeleton PRIVATE pybind11::module)
