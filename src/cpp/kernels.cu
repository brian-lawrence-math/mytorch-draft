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
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t* shape, size_t* strides, size_t dim) {
	size_t result = 0;
	for(size_t d = dim; d-- > 0; ) {
		result += (flat_idx % *(shape + d)) * *(strides + d);
		flat_idx /= *(shape + d);
	}
	return result;
}

__global__ void add_contiguous(float* a, float* b, float* res, size_t len) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len) {
		res[idx] = a[idx] + b[idx];
	}
	return;
}

__global__ void add(float* tensor_a, float* tensor_b, float* tensor_res, size_t* shape, size_t* a_strides, size_t* b_strides, size_t dim) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(shape, dim);

	if (idx < num_entries) {
		size_t a_idx = flat_idx_to_raw_idx_device(idx, shape, a_strides, dim);
		size_t b_idx = flat_idx_to_raw_idx_device(idx, shape, b_strides, dim);

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
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		add<<<1, num_entries>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), b->strides_.data(), a->dim_);
	} else {
		size_t n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		add<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), b->strides_.data(), a->dim_);
	}
}


