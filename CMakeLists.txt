cmake_minimum_required(VERSION 3.18)

include(FetchContent)

# Order matter on the CMake file!1
# 2026-04-07 Necessary for mafalda-2. Needs to be first!
set(CMAKE_CUDA_HOST_COMPILER $ENV{CC})

# 2026-04-07 Change the order of this 2 commands, so it can be compiled in mafalda-2
set(CMAKE_CUDA_ARCHITECTURES 70)
project(sscCdi LANGUAGES CXX CUDA)

#add_definitions(-w)
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.13.0
)

FetchContent_MakeAvailable(spdlog)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --default-stream per-thread")

include_directories(
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

include_directories(
    cuda/inc
    cuda/inc/common
    cuda/inc/common10
    cuda/inc/ptycho
    cuda/inc/planewave
)

# suppress annoying warning to allow header function definition
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20050")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler -use_fast_math --ptxas-options=-v -c --compiler-options -fPIC -G")
set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB_RECURSE CDI_SRC cuda/src/**/*.c*)

add_library(ssccdi SHARED ${CDI_SRC})
target_link_libraries(ssccdi PUBLIC ${CUDA_LIBRARIES} spdlog CUDA::cufft CUDA::cudart m pthread)

install(TARGETS ssccdi LIBRARY DESTINATION ${PROJECT_NAME}/lib)
