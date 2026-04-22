cmake_minimum_required(VERSION 3.15)

project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FIND_PYTHON ON)
# generate PTX intermediate representation during compilation
set(CMAKE_CUDA_ARCHITECTURES "86-real;86-virtual")
enable_language(CUDA)

set(MYTORCH_VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")

if(NOT EXISTS "${MYTORCH_VENV_PYTHON}")
  message(FATAL_ERROR "Expected virtualenv python at ${MYTORCH_VENV_PYTHON}")
endif()

# Tell cmake where to find pybind11
execute_process(COMMAND ${MYTORCH_VENV_PYTHON} -m pybind11 --cmakedir
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(COMPILE_SOURCES src/cpp/tensor.cpp src/cpp/kernels.cu src/cpp/utils.cu src/cpp/pointwise.cu src/cpp/reduce.cu)

add_library(mytorch_core STATIC ${COMPILE_SOURCES})

pybind11_add_module(_core MODULE src/cpp/bindings.cpp) 

add_executable(bench tests/cpp/bench_matmul.cpp)
add_executable(test tests/cpp/test_tensor.cpp)
add_executable(debug tests/cpp/debug_matmul.cpp ${COMPILE_SOURCES})

foreach(tgt IN ITEMS _core bench test)
	target_link_libraries(${tgt} PRIVATE mytorch_core)
endforeach()

foreach(tgt IN ITEMS mytorch_core debug)
	target_link_libraries(${tgt} PUBLIC CUDA::cudart CUDA::curand CUDA::cublas)
	set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

foreach(tgt IN ITEMS bench debug test)
	target_include_directories(${tgt} PRIVATE src/cpp)
endforeach()

target_compile_options(debug PRIVATE
    -g $<$<COMPILE_LANGUAGE:CUDA>:-G>
)

target_compile_options(bench PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

#/opt/cuda/targets/x86_64-linux
set_target_properties(mytorch_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/python/mytorch)

install(TARGETS _core DESTINATION .)
