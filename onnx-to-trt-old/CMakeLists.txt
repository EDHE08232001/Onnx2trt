cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(trt_inference LANGUAGES CXX)

# Specify the C++ standard to be used
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Load environment variables using dotenv.cmake
include(dotenv.cmake)

# Debugging: Display initial cuDNN-related configuration values
message(STATUS "USE_CUDNN initially set to: ${USE_CUDNN}")
message(STATUS "CAFFE2_USE_CUDNN initially set to: ${CAFFE2_USE_CUDNN}")

# Enable cuDNN usage for this project
set(USE_CUDNN 1)
set(CAFFE2_USE_CUDNN 1)

# OpenCV Configuration
# Set OpenCV directories explicitly
set(OpenCV_INCLUDE_DIRS "C:/DevelopmentTools/OpenCV/install/include")
set(OpenCV_LIB_DIR "C:/DevelopmentTools/OpenCV/install/x64/vc16/lib")
set(OpenCV_BIN_DIR "C:/DevelopmentTools/OpenCV/install/x64/vc16/bin")
set(OpenCV_DIR "C:/DevelopmentTools/OpenCV/install/") # where OpenCVConfig.cmake file is located

# Locate OpenCV using the configuration file
find_package(OpenCV REQUIRED)

# Check if OpenCV was found and display its details
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV library directories: ${OpenCV_LIB_DIR}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
link_directories(${OpenCV_LIB_DIR})

# Set the appropriate OpenCV library based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_LIBS opencv_world470d)  # Debug version of opencv_world
else()
    set(OpenCV_LIBS opencv_world470)   # Release version of opencv_world
endif()

# End of OpenCV Configuration

# Finds Python libraries and sets them as required
find_package(Python REQUIRED)
if(NOT Python_FOUND)
    message(FATAL_ERROR "Python not found.")
endif()

# Finds Torch libraries and sets them as required
set(Torch_DIR "C:/Users/Edwar/AppData/Local/Programs/Python/Python310/Lib/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Torch not found.")
endif()
message(STATUS "TORCH_INCLUDE_DIRS set to: ${TORCH_INCLUDE_DIRS}")

# Finds CUDA libraries and displays their details
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message(STATUS "CUDA Version: ${CUDA_VERSION}")
  message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
endif()

# Manually specify the cuDNN library directory
set(CUDNN_LIB_DIR "C:/Program Files/NVIDIA/CUDNN/v8.9.7/lib/x64")

# Locate cuDNN libraries manually if required
set(CUDNN_FOUND 1)
if(CUDNN_FOUND)
  file(GLOB_RECURSE CUDNN_LIBRARIES "${CUDNN_LIB_DIR}/*.lib")
  message(STATUS "cuDNN found")
else()
  message(FATAL_ERROR "cuDNN not found.")
endif()

# Setting PyTorch-specific C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# To check if user wants debug or release mode, default to Release if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
  message(STATUS "Build type not specified: defaulting to Release.")
endif()

# Append compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif()

# Add linker flags to avoid dependency issues with default libraries on Windows
set(CMAKE_EXE_LINKER_FLAGS "/NODEFAULTLIB:LIBCMT")

# Include directories for additional dependencies (e.g., CUDA, TensorRT)
include_directories(
    ${OpenCV_INCLUDE_DIRS} # Include OpenCV headers
    "C:/Program Files/NVIDIA/TensorRT-8.5.3.1/include" # TensorRT headers
    $ENV{CUDA_INCLUDE_DIR} # CUDA headers
)

# Add library directories for linker to find additional libraries
link_directories(
    ${OpenCV_LIB_DIR} # OpenCV libraries
    "C:/Program Files/NVIDIA/TensorRT-8.5.3.1/lib" # TensorRT libraries
    $ENV{CUDA_LIB_DIR} # CUDA libraries
)

# DEBUG: To verify OpenCV and Torch libraries are found correctly
message("OpenCV_LIBS: ${OpenCV_LIBS}")
message("Torch Libraries: ${TORCH_LIBRARIES}")

# Define the executable and its source files
add_executable(trt_inference 
    main.cpp
    TRTGeneration.cpp
    TRTInference.cpp
    ImageProcessingUtil.cpp
)

# Link the necessary libraries to the executable
# Linking opencv_world instead of individual modules (e.g., opencv_core, opencv_imgproc)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_LIBS opencv_world470d) # Debug version
else()
    set(OpenCV_LIBS opencv_world470) # Release version
endif()

target_link_libraries(trt_inference 
    ${OpenCV_LIBS} # Use the monolithic OpenCV library
    ${TORCH_LIBRARIES} # PyTorch libraries
    nvonnxparser # TensorRT ONNX parser
    nvparsers # TensorRT parsers
    nvinfer # TensorRT inference engine
    cudart # CUDA runtime library
    ${CUDNN_LIBRARIES} # cuDNN libraries
)

# Set additional properties for the executable
set_property(TARGET trt_inference PROPERTY CXX_STANDARD 17)
target_compile_features(trt_inference PRIVATE cxx_std_14)

# For Windows builds: copy Torch DLLs to the output directory
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET trt_inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:trt_inference>)
endif()

# DEBUG: Display final configuration for cuDNN usage
message(STATUS "USE_CUDNN finally is: ${USE_CUDNN}")
message(STATUS "CAFFE2_USE_CUDNN finally set to: ${CAFFE2_USE_CUDNN}")

# Commands to run CMake file:
# First, make a build folder and run the following commands inside the build folder
# Debug   Mode: cmake -DCMAKE_PREFIX_PATH=C:\Users\Edwar\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\share\cmake -DCMAKE_BUILD_TYPE=Debug .. ; cmake --build . --config Debug
# Release Mode: cmake -DCMAKE_PREFIX_PATH=C:\Users\Edwar\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\share\cmake -DCMAKE_BUILD_TYPE=Release .. ; cmake --build . --config Release