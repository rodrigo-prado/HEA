cmake_minimum_required(VERSION 3.8)
project(HEA LANGUAGES CXX CUDA)

include_directories(include)
find_package(CUDA)
find_package(OpenMP)
if(OPENMP_FOUND)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp -lgomp")

endif()

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -O3 -fopenmp -std=c++11")

add_executable(HEA_cuda hea.cu)

set_target_properties(HEA_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET HEA_cuda PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
