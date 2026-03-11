GPP = g++
CXXFLAGS = -O3 -Wall -std=c++17 -Isrc/cpp
DEBUG_FLAGS = -g -O0 -Wall -std=c++17
SHARED_FLAGS = -shared -fPIC

PYBIND11_INCLUDES := $(shell uv run python -m pybind11 --includes)
CUDA_DIR := /opt/cuda/targets/x86_64-linux
CUDA_INCLUDES := -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib -lcudart
# EXT_SUFFIX includes the .
# EXT_SUFFIX = .cpython-314-x86_64-linux-gnu.so
EXT_SUFFIX := $(shell uv run python-config --extension-suffix)

SRC_DIR = src/cpp
TEST_DIR = tests/cpp
PYTHON_CORE = src/python/mytorch/_core

TARGET = $(TEST_DIR)/test_tensor
SRC = $(SRC_DIR)/tensor.cpp $(TEST_DIR)/test_tensor.cpp
BINDING_SRC = $(SRC_DIR)/bindings.cpp
LOCAL_INCLUDES = $(SRC_DIR)/tensor.h $(SRC_DIR)/cuda_utils.h

all: $(TARGET)

$(TARGET): $(SRC) $(LOCAL_INCLUDES)
	$(GPP) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $(SRC) -o $(TARGET)

debug:
	$(GPP) $(DEBUG_FLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $(SRC) -o $(TARGET)

python: $(SRC) $(BINDING_SRC) $(LOCAL_INCLUDES)
	$(GPP) $(CXXFLAGS) $(SHARED_FLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $(SRC) $(BINDING_SRC) -o $(PYTHON_CORE)$(EXT_SUFFIX)

clean:
	rm -f $(TARGET) *.o *$(EXT_SUFFIX)


