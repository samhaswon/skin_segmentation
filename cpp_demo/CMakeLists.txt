cmake_minimum_required(VERSION 3.24)

project(skin_segmentation_cpp_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-gpu-1.25.0"
    CACHE PATH "Path to an ONNX Runtime release directory")

if(WIN32)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
else()
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
endif()

set(BUILD_LIST core,imgproc,imgcodecs CACHE STRING "OpenCV modules to build" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable OpenCV tests" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "Disable OpenCV perf tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable OpenCV examples" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "Disable OpenCV apps" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "Disable OpenCV G-API" FORCE)
set(BUILD_opencv_python_bindings_generator OFF CACHE BOOL "Disable Python generator" FORCE)
set(BUILD_opencv_java_bindings_generator OFF CACHE BOOL "Disable Java generator" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "Disable Java" FORCE)
set(BUILD_PROTOBUF OFF CACHE BOOL "Disable Protobuf" FORCE)
set(WITH_IPP OFF CACHE BOOL "Disable IPP download" FORCE)
set(WITH_QUIRC OFF CACHE BOOL "Disable quirc" FORCE)
set(ENABLE_LIBJPEG_TURBO_SIMD OFF CACHE BOOL "Disable libjpeg-turbo SIMD for demo build" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../opencv"
                 "${CMAKE_CURRENT_BINARY_DIR}/opencv-build"
                 EXCLUDE_FROM_ALL)

add_executable(cpp_demo
    main.cpp
    segmentationsession.cpp
)

target_include_directories(cpp_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/modules/core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/modules/imgproc/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../opencv/modules/imgcodecs/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(cpp_demo PRIVATE
    onnxruntime
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

if(MSVC)
    target_compile_options(cpp_demo PRIVATE /W4)
else()
    target_compile_options(cpp_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(cpp_demo PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
endif()
