cmake_minimum_required(VERSION 3.15)

project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FIND_PYTHON ON)

# Tell cmake where to find pybind11
execute_process(COMMAND .venv/bin/python -m pybind11 --cmakedir
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_PREFIX_PATH $(pybind11_DIR))

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/cpp/tensor.cpp src/cpp/bindings.cpp)
install(TARGETS _core DESTINATION .)
