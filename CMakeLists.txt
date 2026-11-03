cmake_minimum_required(VERSION 3.21)
project(onnx2trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # lets VS Code's C/C++ extension find headers

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies. Each can be located through a CMake variable (-D...) or an
# environment variable of the same name; see README.md.
#   CUDA      CUDA_PATH / CUDAToolkit_ROOT
#   TensorRT  TENSORRT_ROOT   (folder containing include/ and lib/)
#   OpenCV    OpenCV_DIR      (folder containing OpenCVConfig.cmake)
# ---------------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

# TensorRT
set(TENSORRT_ROOT "" CACHE PATH "TensorRT install folder (contains include/ and lib/)")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT install folder" FORCE)
endif()
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS "${TENSORRT_ROOT}" PATH_SUFFIXES include)
find_library(TENSORRT_NVINFER_LIB nvinfer
  HINTS "${TENSORRT_ROOT}" PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_ONNXPARSER_LIB nvonnxparser
  HINTS "${TENSORRT_ROOT}" PATH_SUFFIXES lib lib64 lib/x64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_ONNXPARSER_LIB)
  message(FATAL_ERROR
    "TensorRT not found. Set TENSORRT_ROOT to the folder containing include/ and lib/, e.g.\n"
    "  cmake -DTENSORRT_ROOT=/path/to/TensorRT-8.5.3.1 ...")
endif()

# This code uses the TensorRT 8 binding API (enqueueV2, getBindingDimensions, ...),
# which was removed in TensorRT 10.
file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line REGEX "define NV_TENSORRT_MAJOR")
string(REGEX MATCH "[0-9]+" TENSORRT_MAJOR "${_trt_major_line}")
if(TENSORRT_MAJOR AND TENSORRT_MAJOR GREATER 8)
  message(FATAL_ERROR "TensorRT ${TENSORRT_MAJOR}.x found, but this project needs TensorRT 8.x (tested with 8.5.3.1).")
endif()
message(STATUS "TensorRT ${TENSORRT_MAJOR}.x: ${TENSORRT_NVINFER_LIB}")

add_library(TensorRT INTERFACE)
target_include_directories(TensorRT INTERFACE "${TENSORRT_INCLUDE_DIR}")
target_link_libraries(TensorRT INTERFACE "${TENSORRT_NVINFER_LIB}" "${TENSORRT_ONNXPARSER_LIB}")

# OpenCV (stock build is enough, no contrib modules needed)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
message(STATUS "OpenCV ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(onnx2trt
  src/main.cpp
  src/image_utils.cpp
  src/trt_generation.cpp
  src/trt_inference.cpp
)

target_link_libraries(onnx2trt PRIVATE TensorRT CUDA::cudart ${OpenCV_LIBS})
target_include_directories(onnx2trt PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
  target_compile_options(onnx2trt PRIVATE /W3 /utf-8)
else()
  target_compile_options(onnx2trt PRIVATE -Wall -Wextra)
endif()

# On Windows the TensorRT and OpenCV DLLs must sit next to the .exe (or be on PATH).
# Copy them after every build so the program runs straight from VS Code.
if(WIN32)
  get_filename_component(_trt_lib_dir "${TENSORRT_NVINFER_LIB}" DIRECTORY)
  file(GLOB _trt_dlls "${_trt_lib_dir}/*.dll")
  if(_trt_dlls)
    add_custom_command(TARGET onnx2trt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_trt_dlls} "$<TARGET_FILE_DIR:onnx2trt>"
      COMMAND_EXPAND_LISTS)
  endif()
  add_custom_command(TARGET onnx2trt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:onnx2trt>>,copy_if_different,true>
            $<TARGET_RUNTIME_DLLS:onnx2trt> "$<TARGET_FILE_DIR:onnx2trt>"
    COMMAND_EXPAND_LISTS)
endif()
