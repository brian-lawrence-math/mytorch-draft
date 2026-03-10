#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "tensor.h"
#include "cuda_utils.h"

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
	case Device::GPU: {
		float result = 0.0; // block compiler warning "result may be returned uninitialized"
		CUDA_CHECK(cudaMemcpy(&result, p, sizeof(float), cudaMemcpyDeviceToHost));
		return result;
					  }
	default:
		throw std::runtime_error("Invalid device; code should be unreachable.");
	}
}

void write_float(float* p, float val, Device dev) {
	switch(dev) {
	case Device::CPU:
		*p = val;
		break;
	case Device::GPU:
		CUDA_CHECK(cudaMemcpy(p, &val, sizeof(float), cudaMemcpyHostToDevice));
		break;
	}
}

void copy_floats(float* dst, Device dst_dev, float* src, Device src_dev, size_t n) {
	size_t n_bytes = n * sizeof(float);
	switch(dst_dev) {
	case Device::CPU:
		switch(src_dev) {
		case Device::CPU:
			std::memcpy(dst, src, n_bytes);
			break;
		case Device::GPU:
			CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToHost));
			break;
		}
		break;
	case Device::GPU:
		switch(src_dev) {
		case Device::CPU:
			CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToDevice));
			break;
		case Device::GPU:
			CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToDevice));
			break;
		}
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

	FloatBlock* clone(Device new_dev) {
		// debug only:
		FloatBlock* new_block = new FloatBlock(size, new_dev);
		copy_floats(new_block->data, new_dev, this->data, this->dev, size);
		
		float val;
		CUDA_CHECK(cudaMemcpy(&val, new_block->data, sizeof(float), cudaMemcpyDeviceToHost));

		return new_block;
	}
};

// Index into a tensor
// ===================== Constructors ====================
FloatTensor::FloatTensor(std::shared_ptr<FloatBlock> block, size_t dim, std::vector<size_t> shape, size_t offset, std::vector<size_t> strides)
	: block_(block), dim_(dim), shape_(shape), offset_(offset), strides_(strides) {}

// since FloatBlock cannot be copied,
// need to explicitly write copy constructors for FloatTensor
FloatTensor::FloatTensor(const FloatTensor& other) 
	: block_(other.block_), dim_(other.dim_), shape_(other.shape_), offset_(other.offset_), strides_(other.strides_) { }

FloatTensor FloatTensor::zeros_1d(size_t size) {
	FloatBlock* block_raw = FloatBlock::zeros_cpu(size);
	std::shared_ptr<FloatBlock> block(block_raw);

	size_t dim = 1;
	std::vector<size_t> shape{size};
	size_t offset = 0;
	std::vector<size_t> strides{1};
	return FloatTensor{block, dim, shape, offset, strides};
}

// =============== Indexing and shape management ==================
float FloatTensor::get_raw_idx(size_t idx) {
	return this->block_->get_raw_idx(idx);
}

void FloatTensor::set_raw_idx(size_t idx, float val) {
	this->block_->set_raw_idx(idx, val);
}

Index FloatTensor::validate_and_normalize_idx(const Index& idx) {
	if (idx.coords.size() != this->dim_) {
		throw std::length_error("Length of index must match dimension of tensor.");
	}
	std::vector<ssize_t> result{};
	for (size_t d = 0; d < this->dim_; d++) {
		if (idx.coords[d] >= (ssize_t)this->shape_[d] || idx.coords[d] < -(ssize_t)this->shape_[d]) {
			throw std::out_of_range("Index into tensor out of range.");
		}
		size_t norm_coord = idx.coords[d] >= 0 ? idx.coords[d] : this->shape_[d] + idx.coords[d];
		result.push_back(norm_coord);
	}
	return Index{result};
}

size_t FloatTensor::idx_to_raw_idx(Index idx) {
	idx = validate_and_normalize_idx(idx);
	ssize_t raw_idx = this->offset_;
	for (size_t d = 0; d < this->dim_; d++) {
		raw_idx += idx.coords[d] * this->strides_[d];
	}
	return (size_t)raw_idx;
}

float FloatTensor::get_idx(std::vector<ssize_t> coords) {
	Index idx = Index{coords};
	size_t raw_idx = idx_to_raw_idx(idx);
	return get_raw_idx(raw_idx);
}

void FloatTensor::set_idx(std::vector<ssize_t> coords, float val) {
	Index idx = Index{coords};
	size_t raw_idx = idx_to_raw_idx(idx);
	set_raw_idx(raw_idx, val);
}

// A rather hacky function.
// Resets shapes and strides to give a contiguous view of the underlying base FloatBlock.
// If 'this' currently has non-contiguous shape and strides (i.e. is a view),
// they are discarded.
// New_shape must be compatible with the size of the base FloatBlock.
void FloatTensor::base_and_reshape(std::vector<size_t> new_shape) {
	// Compute strides and offsets
	// new_shape will be validated at the end so we only have to compute the product once
	size_t new_dim = new_shape.size();
	size_t new_offset = 0;
	std::vector<size_t> new_strides(new_dim);
	size_t cml_prod = 1;
	for (size_t i = new_dim; i -- > 0; ) {
		new_strides[i] = cml_prod;
		cml_prod *= new_shape[i];
	}
	// validate
	if ((size_t)cml_prod != this->block_->size) {
		throw std::invalid_argument("New shape must match size of underlying base FloatBlock.");
	}
	
	// great, all set, now reassign everything
	this->dim_ = new_dim;
	this->shape_ = new_shape;
	this->offset_ = new_offset;
	this->strides_ = new_strides;
}

// =================== Memory management ==========================
Device FloatTensor::dev_() {
	return this->block_->dev;
}

FloatTensor FloatTensor::clone() {
	std::shared_ptr<FloatBlock> new_block(this->block_->clone(this->dev_()));
	return FloatTensor(new_block, dim_, shape_, offset_, strides_);
}

void FloatTensor::move_to_device(Device dev) {
	if (dev == this->dev_()) {
		return;
	}

	// copy the memory
	std::shared_ptr<FloatBlock> new_block(this->block_->clone(dev));
	this->block_ = new_block;

	// now the old this->block will lose one reference,
	// and if necessary it will be automatically deallocated.
	// Metadata remains unchanged.
	return;
}
