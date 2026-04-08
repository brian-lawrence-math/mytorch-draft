cmake_minimum_required(VERSION 3.15)

project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FIND_PYTHON ON)
enable_language(CUDA)

# Tell cmake where to find pybind11
execute_process(COMMAND .venv/bin/python -m pybind11 --cmakedir
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(_core MODULE src/cpp/tensor.cpp src/cpp/bindings.cpp src/cpp/kernels.cu)

add_executable(bench src/cpp/tensor.cpp src/cpp/kernels.cu tests/cpp/bench_matmul.cpp)
add_executable(debug src/cpp/tensor.cpp src/cpp/kernels.cu tests/cpp/debug_matmul.cpp)
add_executable(test src/cpp/tensor.cpp src/cpp/kernels.cu tests/cpp/test_tensor.cpp)

target_link_libraries(_core PRIVATE CUDA::cudart)
target_link_libraries(bench PRIVATE CUDA::cudart)
target_link_libraries(debug PRIVATE CUDA::cudart)
target_link_libraries(test PRIVATE CUDA::cudart)
target_link_libraries(_core PRIVATE CUDA::curand CUDA::cublas)
target_link_libraries(bench PRIVATE CUDA::curand CUDA::cublas)
target_link_libraries(debug PRIVATE CUDA::curand CUDA::cublas)
target_link_libraries(test PRIVATE CUDA::curand CUDA::cublas)

target_compile_options(debug PRIVATE
    -g $<$<COMPILE_LANGUAGE:CUDA>:-G>
)

target_compile_options(bench PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

#/opt/cuda/targets/x86_64-linux
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/mytorch)

set_target_properties(_core PROPERTIES CUDA_ARCHITECTURES 86)
set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES 86)
set_target_properties(debug PROPERTIES CUDA_ARCHITECTURES 86)
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES 86)

target_include_directories(bench PRIVATE src/cpp)
target_include_directories(debug PRIVATE src/cpp)
target_include_directories(test PRIVATE src/cpp)

install(TARGETS _core DESTINATION .)
