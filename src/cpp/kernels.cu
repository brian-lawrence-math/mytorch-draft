#include <stdexcept>
#include <vector>

#include "tensor.h"
#include "cuda_utils.h"


// Hard-coded constant specific to the RTX 3050
#define MAX_THREADS_PER_BLOCK (1024)


// ========================== Helper methods for indexing =====================
// Modified from tensor.cpp to run on device (i.e. no vectors)
__device__ size_t product_device(size_t* vals, size_t n) {
	size_t cml_prod = 1;
	for(size_t i = 0; i < n; i++) {
		cml_prod *= *(vals + i);
	}
	return cml_prod;
}

// shape is an array size_t[dim]
// strides is another array size_t[dim] of the same size
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t* shape, ssize_t* strides, size_t offset, size_t dim) {
	ssize_t result = offset;
	for(size_t d = dim; d-- > 0; ) {
		result += (flat_idx % *(shape + d)) * *(strides + d);
		flat_idx /= *(shape + d);
	}
	return (size_t)result;
}

__global__ void add_contiguous(float* a, float* b, float* res, size_t len) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len) {
		res[idx] = a[idx] + b[idx];
	}
	return;
}

__global__ void add(float* tensor_a, float* tensor_b, float* tensor_res, size_t* shape, ssize_t* a_strides, size_t a_offset, ssize_t* b_strides, ssize_t  b_offset, size_t dim) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(shape, dim);

	if (idx < num_entries) {
		size_t a_idx = flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
		size_t b_idx = flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

		float val = *(tensor_a + a_idx) + *(tensor_b + b_idx);
		*(tensor_res + idx) = val;
	}
	return;
}

__host__ void launch_add_contiguous(float* a, float* b, float* res, size_t len) {
	if (len < MAX_THREADS_PER_BLOCK) {
		add_contiguous<<<1, len>>>(a, b, res, len);
	} else {
		size_t n_blocks = (len + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		add_contiguous<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(a, b, res, len);
	}
	return;
}

__host__ void launch_add(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	add<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}


