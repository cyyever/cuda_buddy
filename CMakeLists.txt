cmake_minimum_required(VERSION 3.17)

project(CUDABuddyAllocator LANGUAGES CXX)

include(cmake/all.cmake)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(CUDABuddyAllocator ${SOURCES})

find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)

target_link_libraries(CUDABuddyAllocator PRIVATE CUDA::cudart CUDA::cudart_static)
target_link_libraries(CUDABuddyAllocator PRIVATE spdlog::spdlog_header_only)

# test
add_subdirectory(test)

# install lib
install(
  TARGETS CUDABuddyAllocator
  EXPORT CUDABuddyAllocatorConfig
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib COMPONENT library
  INCLUDES
  DESTINATION include)

install(
  DIRECTORY src/
  DESTINATION include/cuda_buddy
  FILES_MATCHING
  PATTERN "*.hpp")

# install cmake config file
install(EXPORT CUDABuddyAllocatorConfig DESTINATION lib/cmake/CUDABuddyAllocator)
