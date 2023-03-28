cmake_minimum_required(VERSION 3.5)

project(my_project)

include_directories(../include)
include_directories(../cxxopts/include)

# Process Abseil's CMake build system
add_subdirectory(abseil-cpp)

add_executable(basic AdjacencyFlatHashSet.cpp)

# Declare dependency on the absl::strings library
target_link_libraries(basic absl::flat_hash_map)
