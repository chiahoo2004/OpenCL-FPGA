# project
cmake_minimum_required(VERSION 2.8)
project(Enhancement)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake")

# enable C++11
if(NOT CMAKE_CXX_COMPILER_ID EQUAL "MSVC")
	add_definitions(--std=c++11)
endif()

# libraries
find_package(DevIL REQUIRED)
find_package(Glog REQUIRED)
include_directories(${IL_INCLUDE_DIR})

add_executable(enhancement
	bilateralfilter.cpp
	enhancement.cpp
	filter.cpp
	gaussian.cpp
	gaussianfilter.cpp
	guidedfilter.cpp
	utilities.cpp
)
target_link_libraries(enhancement ${IL_LIBRARIES} ${GLOG_LIBRARIES})
