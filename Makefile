GPP = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC

PYBIND11_INCLUDES := $(shell uv run python -m pybind11 --includes)
CUDA_DIR := /opt/cuda/targets/x86_64-linux
CUDA_INCLUDES := -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib -lcudart
EXT_SUFFIX := $(shell uv run python-config --extension-suffix)

TARGET = test_tensor
SRC = tensor.cpp tensor.h test_tensor.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(GPP) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(CUDA_INCLUDES) $^ -o $@

clean:
	rm -f $(TARGET) *.o

