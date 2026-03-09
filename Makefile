GPP = g++
CXXFLAGS = -O3 -Wall -std=c++17
DEBUG_FLAGS = -g -O0 -Wall -std=c++17
SHARED_FLAGS = -shared -fPIC

PYBIND11_INCLUDES := $(shell uv run python -m pybind11 --includes)
CUDA_DIR := /opt/cuda/targets/x86_64-linux
CUDA_INCLUDES := -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib -lcudart
EXT_SUFFIX := $(shell uv run python-config --extension-suffix)

TARGET = test_tensor
SRC = tensor.cpp test_tensor.cpp
LOCAL_INCLUDES = tensor.h cuda_utils.h

all: $(TARGET)

$(TARGET): $(SRC) $(LOCAL_INCLUDES)
	$(GPP) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $(SRC) -o $(TARGET)

debug:
	$(GPP) $(DEBUG_FLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

