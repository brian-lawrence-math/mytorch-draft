#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace py = pybind11;

// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
#define CUDA_CHECK(expr) do {										\
	cudaError_t result = expr;										\
	if (result != cudaSuccess) {									\
		fprintf(													\
				stderr,												\
				"CUDA runtime error: %s:%i:%d = %s\n",				\
				__FILE__,											\
				__LINE__,											\
				result,												\
				cudaGetErrorString(result));						\
	}																\
	throw std::runtime_error("CUDA error.");						\
} while (0);															\

enum class Device {
	CPU,
	GPU
};

float* alloc_float(size_t n, Device dev) {
	void* p;
	switch(dev) {
	case Device::CPU:
		p = ::operator new(n * sizeof(float));
		break;
	case Device::GPU:
		p = nullptr;
		CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
		break;
	default:
		throw std::runtime_error("Invalid device; code should be unreachable.");
	}
	return static_cast<float*>(p);
}

void dealloc_float(float* p, Device dev) {
	switch(dev) {
	case Device::CPU:
		::operator delete(p);
		return;
	case Device::GPU:
		CUDA_CHECK(cudaFree(p));
		return;
	}
}

float read_float(float* p, Device dev) {
	switch(dev) {
	case Device::CPU:
		return *p;
	case Device::GPU:
		float result;
		CUDA_CHECK(cudaMemcpy(&result, p, sizeof(float), cudaMemcpyDeviceToHost));
		return result;
	default:
		throw std::runtime_error("Invalid device; code should be unreachable.");
	}
}

void write_float(float* p, float val, Device dev) {
	switch(dev) {
	case Device::CPU:
		*p = val;
	case Device::GPU:
		CUDA_CHECK(cudaMemcpy(p, &val, sizeof(float), cudaMemcpyHostToDevice));
	}
}


// Reference-counted pointer to the data underlying a FloatTensor.
// Note that this pointer must be unique: only one FloatBlock should point
// to the same data.
struct FloatBlock {
	float* data;
	size_t size;
	size_t refcount;
	Device dev;

	// disable copying
	FloatBlock(const FloatBlock&) = delete;
	FloatBlock& operator=(const FloatBlock&) = delete;

	static FloatBlock* zeros_cpu(size_t size) {
		Device dev = Device::CPU;
		float* data = alloc_float(size, dev);
		std::fill(data, data+size, 0.0f);
		size_t refcount = 1;
		return new FloatBlock{data, size, refcount, dev};
	}

	void inc_ref() {
		this->refcount++;
	}

	void dec_ref() {
		this->refcount--;
		if(this->refcount == 0) {
			dealloc_float(this->data, this->dev);
			delete this;
		}
		return;
	}

	void validate_raw_idx(size_t idx) {
		if (idx < 0 || idx >= size) {
			throw std::out_of_range("Index out of range.");
		}
	}

	// Get the float at index idx of the underlying memory:
	// no shape/stride logic.
	float get_raw_idx(size_t idx) {
		validate_raw_idx(idx);
		return read_float(data + idx, dev);
	}

	void set_raw_idx(size_t idx, float val) {
		validate_raw_idx(idx);
		write_float(data + idx, val, dev);
	}

	FloatBlock* copy_cpu_to_gpu() {
		Device new_dev = Device::GPU;
		float* new_data = alloc_float(size, new_dev);
		CUDA_CHECK(cudaMemcpy(new_data, data, size * sizeof(float), cudaMemcpyHostToDevice));
		size_t new_size = size;
		size_t new_refcount = 1;
		return new FloatBlock{new_data, new_size, new_refcount, new_dev};
	}

	FloatBlock* copy_gpu_to_cpu() {
		Device new_dev = Device::CPU;
		float* new_data = alloc_float(size, new_dev);
		CUDA_CHECK(cudaMemcpy(new_data, data, size * sizeof(float), cudaMemcpyDeviceToHost));
		size_t new_size = size;
		size_t new_refcount = 1;
		return new FloatBlock{new_data, new_size, new_refcount, new_dev};
	}

};


struct FloatTensor {
	FloatBlock* block;
	size_t dim;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	
	FloatTensor(FloatBlock* block, size_t dim, std::vector<size_t> shape, std::vector<size_t> strides)
		: block(block), dim(dim), shape(shape), strides(strides) {}

	// copy constructor
	FloatTensor(const FloatTensor& other) : block(other.block), dim(other.dim), shape(other.shape), strides(other.strides) {
		block->refcount++;
	}

	// TODO: copy assignment
	//FloatTensor& operator=(const FloatTensor& other) : 

	static FloatTensor zeros_1d(size_t size) {
		FloatBlock* block = FloatBlock::zeros_cpu(size);
		size_t dim = 1;
		std::vector<size_t> shape{size};
		std::vector<size_t> strides{1};
		return FloatTensor{block, dim, shape, strides};
	}

	Device dev() {
		return block->dev;
	}
};
