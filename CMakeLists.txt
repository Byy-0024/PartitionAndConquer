cmake_minimum_required(VERSION 3.1)
project(Batch_Intersection
LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(main src/Graph.cpp src/test.cpp)
add_executable(preprocess src/preprocess.cpp src/Graph.cpp)
add_executable(convert2ligra src/convert2ligra.cpp src/Graph.cpp)
add_executable(convert2edgelist src/convert2edgelist.cpp src/Graph.cpp)