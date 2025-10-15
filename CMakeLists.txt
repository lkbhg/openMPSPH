cmake_minimum_required(VERSION 3.10.0) 
project(kunpengSPH VERSION 0.1.0 LANGUAGES C CXX)

# -----------------------------
# 默认使用 Clang 编译器
# -----------------------------
set(CMAKE_C_COMPILER "clang" CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)

# -----------------------------
# 设置C++标准
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# 配置Release和Debug版本的编译选项
# -----------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    message(STATUS "Building in Release mode with optimizations")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    message(STATUS "Building in Debug mode with debug symbols")
else()
    message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# -----------------------------
# 查找并启用OpenMP
# -----------------------------
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP to continue.")
endif()

# -----------------------------
# 查找并启用Eigen3
# -----------------------------
#find_package(Eigen3 REQUIRED NO_MODULE)
#if(Eigen3_FOUND)
#    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
#else()
#    message(FATAL_ERROR "Eigen3 not found. Please install eigen3-devel to continue.")
#endif()

# -----------------------------
# 设置 include 和 src 目录
# -----------------------------
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

if(NOT EXISTS ${INCLUDE_DIR})
    message(WARNING "Include directory not found: ${INCLUDE_DIR}")
endif()

if(NOT EXISTS ${SRC_DIR})
    message(WARNING "Source directory not found: ${SRC_DIR}")
endif()

include_directories(${INCLUDE_DIR})

# -----------------------------
# 收集源文件
# -----------------------------
file(GLOB_RECURSE SOURCE_FILES 
    ${SRC_DIR}/*.cpp 
    ${SRC_DIR}/*.c 
)

set(MAIN_FILE ${PROJECT_SOURCE_DIR}/main.cpp)
if(NOT EXISTS ${MAIN_FILE})
    message(FATAL_ERROR "Main file not found: ${MAIN_FILE}")
endif()

# -----------------------------
# 添加可执行文件
# -----------------------------
add_executable(kunpengSPH ${MAIN_FILE} ${SOURCE_FILES})

# 链接 OpenMP 和 Eigen
#target_link_libraries(kunpengSPH PRIVATE ${OpenMP_LIBS} Eigen3::Eigen)
target_link_libraries(kunpengSPH PRIVATE ${OpenMP_LIBS})

# -----------------------------
# 设置输出目录
# -----------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)

# -----------------------------
# 显示项目信息
# -----------------------------
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Include directory: ${INCLUDE_DIR}")
message(STATUS "Source directory: ${SRC_DIR}")
message(STATUS "Found ${SOURCE_FILES} source files in src directory")
