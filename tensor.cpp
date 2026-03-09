#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "tensor.h"

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


// RAII: A FloatBlock manages the pointer to an array of floats.
// The FloatBlock owns the memory: when the FloatBlock is destroyed,
// the memory is deallocated.
// Note that this pointer must be unique: only one FloatBlock should point
// to the same data.
struct FloatBlock {
	// Doing manual memory management, rather than a smart pointer,
	// to build understanding.
	// Note this will require separate alloc/dealloc logic for CPU and GPU.
	float* data;
	size_t size;
	Device dev;

private:
	FloatBlock(size_t size, Device dev) {
		float* data = alloc_float(size, dev);
		this->data = data;
		this->size = size;
		this->dev = dev;
	}

public:
	~FloatBlock() {
		dealloc_float(data, dev);
	}

	// disable copying
	FloatBlock(const FloatBlock&) = delete;
	FloatBlock& operator=(const FloatBlock&) = delete;

	static FloatBlock* zeros_cpu(size_t size) {
		FloatBlock* block = new FloatBlock(size, Device::CPU);
		std::fill(block->data, block->data+size, 0.0f);
		return block;
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
		FloatBlock* new_block = new FloatBlock(size, Device::GPU);
		CUDA_CHECK(cudaMemcpy(new_block->data, data, size * sizeof(float), cudaMemcpyHostToDevice));
		return new_block;
	}

	FloatBlock* copy_gpu_to_cpu() {
		FloatBlock* new_block = new FloatBlock(size, Device::CPU);
		CUDA_CHECK(cudaMemcpy(new_block->data, data, size * sizeof(float), cudaMemcpyDeviceToHost));
		return new_block;
	}
};



FloatTensor::FloatTensor(std::shared_ptr<FloatBlock> block, size_t dim, std::vector<size_t> shape, std::vector<size_t> strides)
	: block(block), dim(dim), shape(shape), strides(strides) {}

FloatTensor FloatTensor::zeros_1d(size_t size) {
	FloatBlock* block_raw = FloatBlock::zeros_cpu(size);
	std::shared_ptr<FloatBlock> block(block_raw);

	size_t dim = 1;
	std::vector<size_t> shape{size};
	std::vector<size_t> strides{1};
	return FloatTensor{block, dim, shape, strides};
}

float FloatTensor::get_raw_idx(size_t idx) {
	return this->block->get_raw_idx(idx);
}

void FloatTensor::set_raw_idx(size_t idx, float val) {
	this->block->set_raw_idx(idx, val);
}

Device FloatTensor::dev() {
	return block->dev;
}

