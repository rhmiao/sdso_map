cmake_minimum_required(VERSION 2.4.6)

set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -fopenmp -g -Wall -march=native -msse -msse2 -msse3 -mssse3 -msse4 -ffast-math -mfpmath=sse")

# Uncomment the following line to disable the usage of the popcnt instruction
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKS_NO_POPCNT")

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

OPTION(BUILD_SHARED_LIBS 	"Set to OFF to build static libraries" ON)

# Find Required libraries

#use opencv2.4.10
set(OpenCV_DIR "/home/mrh/workspace/OPENCVLIB/2.4.10")
find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR} NO_DEFAULT_PATH)

find_package(Boost REQUIRED)

# Setting include directories

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
