cmake_minimum_required(VERSION 3.18)

include(FetchContent)

project(sscCdi LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 70)

#add_definitions(-w)
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) #-fPIC

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

find_package(CUDAToolkit)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

include_directories(
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(ssccommons_SOURCE_DIR ./externals/ssc-commons)
add_subdirectory(${ssccommons_SOURCE_DIR})

include_directories(
    cuda/inc
    ${ssccommons_SOURCE_DIR}/cuda/inc
    ${ssccommons_SOURCE_DIR}/cuda/inc/common
    ${ssccommons_SOURCE_DIR}/cuda/inc/common10
)

# suppress annoying warning to allow header function definition
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20050")

file(GLOB_RECURSE CDI_SRC cuda/src/*.c*)

add_library(ssccdi SHARED ${CDI_SRC})
target_link_libraries(ssccdi PUBLIC ${CUDA_LIBRARIES} ssccommons_shared CUDA::cufft CUDA::cudart)
install(TARGETS ssccdi LIBRARY DESTINATION ${PROJECT_NAME}/lib)
set_target_properties(ssccdi PROPERTIES INSTALL_RPATH "$ORIGIN") # look for binaries in same path (to link sscommons)

install(TARGETS ssccommons_wrapper LIBRARY DESTINATION ${PROJECT_NAME}/lib)
install(TARGETS ssccommons_shared LIBRARY DESTINATION ${PROJECT_NAME}/lib)
target_link_libraries(ssccommons_wrapper PUBLIC ssccommons_shared)
set_target_properties(ssccommons_wrapper PROPERTIES INSTALL_RPATH "$ORIGIN") # look for binaries in same path (to link sscommons)
