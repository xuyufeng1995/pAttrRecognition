cmake_minimum_required(VERSION 3.19)

project(inception)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJCTE_SOURCE_DIR})
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# tensorrt
include_directories(/home/xuyufeng/source/TensorRT-7.2.2.3/include/)
link_directories(/home/xuyufeng/source/TensorRT-7.2.2.3/lib/)
# opencv
set(OpenCV_DIR /home/xuyufeng/source/opencv_build/share/OpenCV/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# maxLayer
cuda_add_library(maxLayer SHARED ${PROJECT_SOURCE_DIR}/maxLayer.cu)
target_link_libraries(maxLayer nvinfer cudart ${OpenCV_LIBS})

# affineGridLayer
cuda_add_library(affineGridLayer SHARED ${PROJECT_SOURCE_DIR}/affineGridLayer.cu)
target_link_libraries(affineGridLayer nvinfer cudart ${OpenCV_LIBS})

add_executable(inception ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(inception nvinfer)
target_link_libraries(inception cudart)
target_link_libraries(inception maxLayer)
target_link_libraries(inception affineGridLayer)
target_link_libraries(inception ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
