#include <iostream>
#include <stdexcept>
#include <cstdio>
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
		cml_prod *= vals[i];
	}
	return cml_prod;
}

// shape is an array size_t[dim]
// strides is another array size_t[dim] of the same size
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t* shape, ssize_t* strides, size_t offset, size_t dim) {
	ssize_t result = offset;
	for(size_t d = dim; d-- > 0; ) {
		result += (flat_idx % shape[d]) * strides[d];
		flat_idx /= shape[d];
	}
	return (size_t)result;
}

// =================== Special-case kernels for contiguous tensors ==========
__global__ void add_contiguous(float* a, float* b, float* res, size_t len) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < len) {
		res[idx] = a[idx] + b[idx];
	}
	return;
}

// ======================= Basic tensor arithmetic kernels =============================
__global__ void add(float* tensor_a, float* tensor_b, float* tensor_res, size_t* shape, ssize_t* a_strides, size_t a_offset, ssize_t* b_strides, ssize_t  b_offset, size_t dim) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(shape, dim);

	if (idx < num_entries) {
		size_t a_idx = flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
		size_t b_idx = flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

		float val = tensor_a[a_idx] + tensor_b[b_idx];
		tensor_res[idx] = val;
	}
	return;
}

__global__ void sub(float* tensor_a, float* tensor_b, float* tensor_res, size_t* shape, ssize_t* a_strides, size_t a_offset, ssize_t* b_strides, ssize_t  b_offset, size_t dim) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(shape, dim);

	if (idx < num_entries) {
		size_t a_idx = flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
		size_t b_idx = flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

		float val = tensor_a[a_idx] - tensor_b[b_idx];
		tensor_res[idx] = val;
	}
	return;
}

__global__ void mul(float* tensor_a, float* tensor_b, float* tensor_res, size_t* shape, ssize_t* a_strides, size_t a_offset, ssize_t* b_strides, ssize_t  b_offset, size_t dim) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(shape, dim);

	if (idx < num_entries) {
		size_t a_idx = flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
		size_t b_idx = flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

		float val = tensor_a[a_idx] * tensor_b[b_idx];
		tensor_res[idx] = val;
	}
	return;
}

__global__ void matmul(float* tensor_a, float* tensor_b, float* tensor_res, size_t* a_shape, ssize_t* a_strides, size_t a_offset, size_t* b_shape, ssize_t* b_strides, size_t b_offset, size_t dim) {
	size_t flat_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = product_device(a_shape, dim-2) * a_shape[dim - 2] * b_shape[dim - 1];

	if (flat_thread_idx < num_entries) {
		// Compute a_idx and b_idx for the 0th step of the matmul
		size_t a_idx = a_offset;
		size_t b_idx = b_offset;

		// save this value for later
		size_t res_idx = flat_thread_idx;

		// Process the last two dimensions manually
		size_t last_idx = flat_thread_idx % b_shape[dim - 1];
		b_idx += last_idx * b_strides[dim - 1];

		flat_thread_idx /= b_shape[dim - 1];
		size_t next_to_last_idx = flat_thread_idx % a_shape[dim - 2];
		a_idx += next_to_last_idx * a_strides[dim - 2];

		// The remaining indices are batch indices, just loop through them
		for (size_t d = dim - 2; d-- > 0; ) {
			flat_thread_idx /= a_shape[d + 1];
			size_t curr_d_idx = flat_thread_idx % a_shape[d];
			a_idx += curr_d_idx * a_strides[d];
			b_idx += curr_d_idx * b_strides[d];
		}

		// Now a_idx, b_idx store the first indices to be multiplied.
		// The others will be computed by adding a_step and b_step.
		size_t a_step = a_strides[dim - 1];
		size_t b_step = b_strides[dim - 2];

		// Finally ready for the multiplication loop
		float result = 0;
		for (size_t mul_idx = 0; mul_idx < a_shape[dim - 1]; mul_idx++) {
			result += tensor_a[a_idx] * tensor_b[b_idx];
			a_idx += a_step;
			b_idx += b_step;
		}

		tensor_res[res_idx] = result;
	}
}

struct ContiguousTensor3d_Device {
	float* data; size_t shape[3];
};

// special-case function: batched matmul for contiguous 3d tensors
__global__ void matmul_3d(ContiguousTensor3d_Device a, ContiguousTensor3d_Device b, ContiguousTensor3d_Device res) {
	size_t flat_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	size_t num_entries = res.shape[0] * res.shape[1] * res.shape[2];

	if (flat_thread_idx < num_entries) {
		// save a copy for the final assignment
		size_t res_idx = flat_thread_idx;
		
		// convert flat_thread_idx to idx into res
		size_t z_idx = flat_thread_idx % res.shape[2];
		flat_thread_idx /= res.shape[2];
		size_t y_idx = flat_thread_idx % res.shape[1];
		flat_thread_idx /= res.shape[1];
		size_t x_idx = flat_thread_idx;

		// now the multiplication loop
		// precompute starting idxs and step sizes for a and b
		size_t a_idx = x_idx * a.shape[1] * a.shape[2] + y_idx * a.shape[2];
		size_t b_idx = x_idx * b.shape[1] * b.shape[2] + z_idx;
		size_t a_step = 1;
		size_t b_step = b.shape[2];

		float result = 0;

		for (size_t mul_idx = 0; mul_idx < a.shape[2]; mul_idx ++) {
			result += a.data[a_idx] * b.data[b_idx];
			a_idx += a_step;
			b_idx += b_step;
		}

		res.data[res_idx] = result;
	}
}

// ============================= Launchers ======================================
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
	add<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_sub(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	sub<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_mul(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	mul<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}


__host__ void launch_matmul(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = res->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	matmul<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->shape_.data(), b->strides_.data(), b->offset_, a->dim_);
}


ContiguousTensor3d_Device to_ct3d(FloatTensor* x) {
	return ContiguousTensor3d_Device{x->data_ptr(), {x->shape_[0], x->shape_[1], x->shape_[2]}};
}

__host__ void launch_matmul_3d(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	ContiguousTensor3d_Device a_device = to_ct3d(a);
	ContiguousTensor3d_Device b_device = to_ct3d(b);
	ContiguousTensor3d_Device res_device = to_ct3d(res);

	size_t num_entries = res->numel();

	size_t n_blocks, threads_per_block;
	if (num_entries < MAX_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
		threads_per_block = MAX_THREADS_PER_BLOCK;
	}
	matmul_3d<<<n_blocks, threads_per_block>>>(a_device, b_device, res_device);
}

