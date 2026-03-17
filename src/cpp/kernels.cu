#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <vector>

#include <curand.h>

#include "tensor.h"
#include "cuda_utils.h"


// Hard-coded constant specific to the RTX 3050
#define MAX_THREADS_PER_BLOCK (1024)

// shared mem limits:
//   12288 floats / block
//   25600 floats / SM
//   we only have one block per SM right now anyway

#define TARGET_THREADS_PER_BLOCK (1024)

#define TPB_ROW (32)
#define TPB_COL (32)
#define TILE_ROWS (8)
#define TILE_COLS (8)

// Multiplication tensor(a, b) @ tensor(b, c) 
// means a "multiplication loop" over the intermediate dimension of size b.
// How many of those b entries to load into shared memory at once?
#define MUL_LOOP_TO_LOAD (24)

// make sure shared memory won't overflow
#define MAX_SHARED_FLOATS_PER_BLOCK (12288)
static_assert((TPB_ROW * TILE_ROWS + TPB_COL * TILE_COLS) * MUL_LOOP_TO_LOAD <= MAX_SHARED_FLOATS_PER_BLOCK);

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
		for (size_t loop_idx = 0; loop_idx < a_shape[dim - 1]; loop_idx++) {
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

		for (size_t loop_idx = 0; loop_idx < a.shape[2]; loop_idx ++) {
			result += a.data[a_idx] * b.data[b_idx];
			a_idx += a_step;
			b_idx += b_step;
		}

		res.data[res_idx] = result;
	}
}

__global__ void matmul_tiled(ContiguousTensor3d_Device a, ContiguousTensor3d_Device b, ContiguousTensor3d_Device res) {
	// blocks: (batch, row, col)
	// threads: (1, row, col)
	
	// compute batch
	size_t batch = blockIdx.x;
	float* a_batch_data = a.data + batch * a.shape[1] * a.shape[2];
	float* b_batch_data = b.data + batch * b.shape[1] * b.shape[2];
	float* res_batch_data = res.data + batch * res.shape[1] * res.shape[2];

	// compute rows and cols
	size_t start_row = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_ROWS;
	size_t start_col = (blockIdx.z * blockDim.z + threadIdx.z) * TILE_COLS;

	for (size_t start_loop_idx = 0; start_loop_idx < a.shape[2]; start_loop_idx += MUL_LOOP_TO_LOAD) {
		__syncthreads();

		// collaboratively load:
		// a[batch, start_row: start_row + TILE_ROWS, start_loop_idx: start_loop_idx + MUL_LOOP_TO_LOAD]
		// into
		// a_shared[start_row_shared: start_row_shared + TILE_ROWS, : ]
		__shared__ float a_shared[TPB_ROW * TILE_ROWS][MUL_LOOP_TO_LOAD];

		size_t start_row_shared = threadIdx.y * TILE_ROWS;
		for (size_t i = 0; i < TILE_ROWS && start_row + i < a.shape[1]; i++) {
			for (size_t j = 0; j < MUL_LOOP_TO_LOAD && start_loop_idx + j < a.shape[2]; j++) {
				a_shared[start_row_shared + i][j] = a_batch_data[(start_row + i) * a.shape[2] + start_loop_idx + j];
			}
		}
		

		// b[batch, start_loop_idx: start_loop_idx + MUL_LOOP_TO_LOAD, start_col: start_col + TILE_ROWS]
		__shared__ float b_shared[MUL_LOOP_TO_LOAD][TPB_COL * TILE_COLS];

		size_t start_col_shared = threadIdx.z * TILE_COLS;
		for (size_t i = 0; i < MUL_LOOP_TO_LOAD && start_loop_idx + i < b.shape[1]; i++) {
			for (size_t j = 0; j < TILE_COLS && start_col + j < b.shape[2]; j++) {
				b_shared[i][start_col_shared + j] = b_batch_data[(start_loop_idx + i) * b.shape[2] + (start_col + j)];
			}
		}

		__syncthreads();
		// now that everything has been loaded into a_shared and b_shared,
		// we can do some computation

		for (size_t row = 0; row < TILE_ROWS && start_row + row < a.shape[1]; row++) {
			for (size_t col = 0; col < TILE_COLS && start_col + col < b.shape[2]; col++) {
				float cml_sum = 0.0;

				for (size_t loop_idx = 0; loop_idx < MUL_LOOP_TO_LOAD && start_loop_idx + loop_idx < a.shape[2]; loop_idx++) {
					cml_sum += a_shared[start_row_shared + row][loop_idx] * b_shared[loop_idx][start_col_shared + col];
				}
			
				// make the update
				// don't worry about synchronization or atomic add because
				// I'm the only thread touching this entry of res
				res_batch_data[(start_row + row) * res.shape[2] + (start_col + col)] += cml_sum;
			}
		}
	}
}

// ============================= Launchers ======================================
__host__ void launch_randn(float* a, size_t len, int seed) {
	curandGenerator_t gen;

	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	CURAND_CALL(curandGenerateNormal(gen, a, len, 0.0, 1.0));

	return;
}

__host__ void launch_add_contiguous(float* a, float* b, float* res, size_t len) {
	if (len < TARGET_THREADS_PER_BLOCK) {
		add_contiguous<<<1, len>>>(a, b, res, len);
	} else {
		size_t n_blocks = (len + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		add_contiguous<<<n_blocks, TARGET_THREADS_PER_BLOCK>>>(a, b, res, len);
	}
	return;
}

__host__ void launch_add(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < TARGET_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		threads_per_block = TARGET_THREADS_PER_BLOCK;
	}
	add<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_sub(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < TARGET_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		threads_per_block = TARGET_THREADS_PER_BLOCK;
	}
	sub<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_mul(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < TARGET_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		threads_per_block = TARGET_THREADS_PER_BLOCK;
	}
	mul<<<n_blocks, threads_per_block>>>(a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}


__host__ void launch_matmul(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	size_t num_entries = res->numel();
	size_t n_blocks, threads_per_block;
	if (num_entries < TARGET_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		threads_per_block = TARGET_THREADS_PER_BLOCK;
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
	if (num_entries < TARGET_THREADS_PER_BLOCK) {
		n_blocks = 1;
		threads_per_block = num_entries;
	} else {
		n_blocks = (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
		threads_per_block = TARGET_THREADS_PER_BLOCK;
	}
	matmul_3d<<<n_blocks, threads_per_block>>>(a_device, b_device, res_device);
}

__host__ void launch_matmul_tiled(FloatTensor* a, FloatTensor* b, FloatTensor* res) {
	ContiguousTensor3d_Device a_device = to_ct3d(a);
	ContiguousTensor3d_Device b_device = to_ct3d(b);
	ContiguousTensor3d_Device res_device = to_ct3d(res);

	size_t grid_dim_x, grid_dim_y, grid_dim_z;
	size_t tpb_y, tpb_z;

	size_t n_rows = a->shape_[1];
	size_t n_cols = b->shape_[2];

	// batches are taken care of by grid_dim,
	// no need to have different batches in the same block
	grid_dim_x = a->shape_[0];

	// rows
	size_t row_threads = (n_rows + TILE_ROWS - 1) / TILE_ROWS;

	if (row_threads > TPB_ROW) {
		tpb_y = TPB_ROW;
		grid_dim_y = (row_threads + TPB_ROW - 1) / TPB_ROW;
	} else {
		tpb_y = row_threads;
		grid_dim_y = 1;
	}

	// cols
	size_t col_threads = (n_cols + TILE_COLS - 1) / TILE_COLS;

	if (col_threads > TPB_COL) {
		tpb_z = TPB_COL;
		grid_dim_z = (col_threads + TPB_COL - 1) / TPB_COL;
	} else {
		tpb_z = col_threads;
		grid_dim_z = 1;
	}

	dim3 blocks(grid_dim_x, grid_dim_y, grid_dim_z);
	dim3 threads(1, tpb_y, tpb_z);
	
	matmul_tiled<<<blocks, threads>>>(a_device, b_device, res_device);
	CUDA_CHECK(cudaGetLastError());
}

