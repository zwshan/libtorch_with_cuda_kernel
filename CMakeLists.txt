cmake_minimum_required (VERSION 3.22)
# Determine the project root directory
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# Define the absolute path to libtorch
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/external/libtorch")

# Set CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# # Add the _GLIBCXX_USE_CXX11_ABI=0 flag
# # This is needed to avoid linker errors with libtorch
# # https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

project(cuda_libtorch_test LANGUAGES CUDA CXX)

include(ProcessorCount)
ProcessorCount(total_cores)
if(NOT DEFINED total_cores OR total_cores EQUAL 0)
    set(total_cores 1)
    set(used_cores 1)
elseif(total_cores GREATER 1)
    math(EXPR used_cores "${total_cores} - 2") # use total_cores-2 if total_cores > 1
endif()
set(ENV{MAKEFLAGS} "-j${used_cores}")
message(STATUS "Building with ${used_cores} out of ${total_cores} available cores")

set(HEADERS
    includes/Libtorch_CUDA_helper.h
    )

set(SOURCES
    src/CUDA_TEST.cpp 
    src/Libtorch_CUDA_helper.cu
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 80 # SELECT THE ARCHITECTURE OF YOUR GPU HERE (Here it is 80 for A100)
    CUDA_STANDARD 17 
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)

# Torch setup
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)

#target compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE -g -G -Xcompiler -Werror)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
  )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()

find_package(CUDAToolkit REQUIRED)

#Check CUDA version
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "This project requires CUDA 11.0 or higher")
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
)

