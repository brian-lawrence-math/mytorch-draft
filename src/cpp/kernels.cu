#include <cassert>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cublas_v2.h"
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
#define TILE_ROWS (4)
#define TILE_COLS (4)

#define TM (4)
#define TN (4)
#define TPBM (32)
#define TPBN (32)
#define BM (TM * TPBM)
#define BN (TN * TPBN)
#define BK (8)

// make sure blockDim.z is a multiple of warp size
static_assert((TPB_ROW % 32 == 0));

// Multiplication tensor(a, b) @ tensor(b, c)
// means a "multiplication loop" over the intermediate dimension of size b.
// How many of those b entries to load into shared memory at once?
#define MUL_LOOP_TO_LOAD (48)

// make sure shared memory won't overflow
#define MAX_SHARED_FLOATS_PER_BLOCK (12288)
static_assert((TPB_ROW * TILE_ROWS + TPB_COL * TILE_COLS) * MUL_LOOP_TO_LOAD <=
              MAX_SHARED_FLOATS_PER_BLOCK);

// ========================== Helper methods for indexing =====================
// Modified from tensor.cpp to run on device (i.e. no vectors)
__device__ size_t product_device(size_t *vals, size_t n) {
  size_t cml_prod = 1;
  for (size_t i = 0; i < n; i++) {
    cml_prod *= vals[i];
  }
  return cml_prod;
}

// shape is an array size_t[dim]
// strides is another array size_t[dim] of the same size
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t *shape,
                                             ssize_t *strides, size_t offset,
                                             size_t dim) {
  ssize_t result = offset;
  for (size_t d = dim; d-- > 0;) {
    result += (flat_idx % shape[d]) * strides[d];
    flat_idx /= shape[d];
  }
  return (size_t)result;
}

// =================== Special-case kernels for contiguous tensors ==========
__global__ void add_contiguous(float *a, float *b, float *res, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < len) {
    res[idx] = a[idx] + b[idx];
  }
  return;
}

// ======================= Basic tensor arithmetic kernels
// =============================
__global__ void is_eq(float *tensor_a, float *tensor_b, size_t *shape,
                      ssize_t *a_strides, size_t a_offset, ssize_t *b_strides,
                      ssize_t b_offset, size_t dim, int *result) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
    size_t b_idx =
        flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

    float a_val = tensor_a[a_idx];
    float b_val = tensor_b[b_idx];

    if (a_val != b_val) {
      atomicCAS(result, 1, 0);
    }
  }
  return;
}

__global__ void contiguous_clone(float *tensor_a, float *tensor_res,
		size_t *shape, ssize_t *a_strides, size_t a_offset, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);
  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);

    float val = tensor_a[a_idx];
    tensor_res[idx] = val;
  }
  return;
}

__global__ void add(float *tensor_a, float *tensor_b, float *tensor_res,
                    size_t *shape, ssize_t *a_strides, size_t a_offset,
                    ssize_t *b_strides, size_t b_offset, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
    size_t b_idx =
        flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

    float val = tensor_a[a_idx] + tensor_b[b_idx];
    tensor_res[idx] = val;
  }
  return;
}

__global__ void sub(float *tensor_a, float *tensor_b, float *tensor_res,
                    size_t *shape, ssize_t *a_strides, size_t a_offset,
                    ssize_t *b_strides, size_t b_offset, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
    size_t b_idx =
        flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

    float val = tensor_a[a_idx] - tensor_b[b_idx];
    tensor_res[idx] = val;
  }
  return;
}

__global__ void mul(float *tensor_a, float *tensor_b, float *tensor_res,
                    size_t *shape, ssize_t *a_strides, size_t a_offset,
                    ssize_t *b_strides, size_t b_offset, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
    size_t b_idx =
        flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

    float val = tensor_a[a_idx] * tensor_b[b_idx];
    tensor_res[idx] = val;
  }
  return;
}

__global__ void matmul(float *tensor_a, float *tensor_b, float *tensor_res,
                       size_t *a_shape, ssize_t *a_strides, size_t a_offset,
                       size_t *b_shape, ssize_t *b_strides, size_t b_offset,
                       size_t dim) {
  size_t flat_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries =
      product_device(a_shape, dim - 2) * a_shape[dim - 2] * b_shape[dim - 1];

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
    for (size_t d = dim - 2; d-- > 0;) {
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
  float *data;
  size_t shape[3];
};

// special-case function: batched matmul for contiguous 3d tensors
__global__ void matmul_3d(ContiguousTensor3d_Device a,
                          ContiguousTensor3d_Device b,
                          ContiguousTensor3d_Device res) {
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

    for (size_t loop_idx = 0; loop_idx < a.shape[2]; loop_idx++) {
      result += a.data[a_idx] * b.data[b_idx];
      a_idx += a_step;
      b_idx += b_step;
    }

    res.data[res_idx] = result;
  }
}

__global__ void matmul_tiled(ContiguousTensor3d_Device a,
                             ContiguousTensor3d_Device b,
                             ContiguousTensor3d_Device res) {
  // blocks: (batch, row, col)
  // threads: (1, row, col)

  // compute batch
  size_t batch = blockIdx.x;
  float *a_batch_data = a.data + batch * a.shape[1] * a.shape[2];
  float *b_batch_data = b.data + batch * b.shape[1] * b.shape[2];
  float *res_batch_data = res.data + batch * res.shape[1] * res.shape[2];

  // which entries is this thread responsible for?
  size_t thread_idx_flat = threadIdx.y + blockDim.y * threadIdx.z;
  size_t tot_threads_this_block = blockDim.y * blockDim.z;

  size_t a_shared_n_entries = blockDim.z * TILE_ROWS * MUL_LOOP_TO_LOAD;
  size_t b_shared_n_entries = MUL_LOOP_TO_LOAD * blockDim.y * TILE_COLS;

  size_t a_shared_row_offset = blockIdx.z * blockDim.z * TILE_ROWS;
  size_t b_shared_col_offset = blockIdx.y * blockDim.y * TILE_COLS;

  for (size_t start_loop_idx = 0; start_loop_idx < a.shape[2];
       start_loop_idx += MUL_LOOP_TO_LOAD) {
    __syncthreads();

    // collaboratively load (1, blockDim.z * TILE_ROWS, MUL_LOOP) values
    // starting from a[batch, blockIdx.z * blockDim.z * TILE_ROWS,
    // start_loop_idx] into a_shared (note blockDim.z <= TPB_ROW always)
    __shared__ float a_shared[TPB_ROW * TILE_ROWS][MUL_LOOP_TO_LOAD];
    static_assert(MUL_LOOP_TO_LOAD % 4 == 0);

    // collaboratively fill up a_shared
    // NOTE: use of float4 here means tensor size must be div by 4
    // To make this work in general, we will need to allocate extra memory
    // when we allocate memory for a tensor
    for (size_t a_shared_idx = 4 * thread_idx_flat;
         a_shared_idx < a_shared_n_entries;
         a_shared_idx += 4 * tot_threads_this_block) {
      size_t a_shared_row = a_shared_idx / MUL_LOOP_TO_LOAD;
      size_t a_shared_col = a_shared_idx % MUL_LOOP_TO_LOAD;
      if (a_shared_row_offset + a_shared_row < a.shape[1] &&
          start_loop_idx + a_shared_col < a.shape[2]) {
        *reinterpret_cast<float4 *>(&a_shared[a_shared_row][4 * a_shared_col]) =
            *reinterpret_cast<float4 *>(
                &a_batch_data[(a_shared_row_offset + a_shared_row) *
                                  a.shape[2] +
                              start_loop_idx + 4 * a_shared_col]);
      }
    }

    // b[batch, start_loop_idx: start_loop_idx + MUL_LOOP_TO_LOAD, start_col:
    // start_col + TILE_ROWS]
    __shared__ float b_shared[MUL_LOOP_TO_LOAD][TPB_COL * TILE_COLS];
    static_assert((TPB_COL * TILE_COLS) % 4 == 0);

    for (size_t b_shared_idx = 4 * thread_idx_flat;
         b_shared_idx < b_shared_n_entries;
         b_shared_idx += 4 * tot_threads_this_block) {
      size_t b_shared_row = b_shared_idx / (blockDim.y * TILE_COLS);
      size_t b_shared_col = b_shared_idx % (blockDim.y * TILE_COLS);
      if (start_loop_idx + b_shared_row < b.shape[1] &&
          b_shared_col_offset + b_shared_col < b.shape[2]) {
        *reinterpret_cast<float4 *>(&b_shared[b_shared_row][b_shared_col]) =
            *reinterpret_cast<float4 *>(
                &b_batch_data[(start_loop_idx + b_shared_row) * b.shape[2] +
                              b_shared_col_offset + b_shared_col]);
      }
    }

    __syncthreads();
    // now that everything has been loaded into a_shared and b_shared,
    // we can do some computation

    for (size_t row = threadIdx.z;
         row < blockDim.z * TILE_ROWS && a_shared_row_offset + row < a.shape[1];
         row += blockDim.z) {
      for (size_t col = threadIdx.y; col < blockDim.y * TILE_COLS &&
                                     b_shared_col_offset + col < b.shape[2];
           col += blockDim.y) {
        // am I doing too much computation in this loop?
        float cml_sum = 0.0;

        for (size_t loop_idx = 0; loop_idx < MUL_LOOP_TO_LOAD &&
                                  start_loop_idx + loop_idx < a.shape[2];
             loop_idx++) {
          cml_sum += a_shared[row][loop_idx] * b_shared[loop_idx][col];
        }

        // make the update
        // don't worry about synchronization or atomic add because
        // I'm the only thread touching this entry of res
        res_batch_data[(a_shared_row_offset + row) * res.shape[2] +
                       (b_shared_col_offset + col)] += cml_sum;
      }
    }
  }
}

__global__ void matmul_tiled_2(ContiguousTensor3d_Device a, ContiguousTensor3d_Device b,
		ContiguousTensor3d_Device res) {
	// blocks: (batch, row, col)
	// threads: (n_threads, 1, 1)  -- I will manage it myself
	
	size_t M = a.shape[1], K = a.shape[2], N = b.shape[2];
	size_t a_row_base = BM * blockIdx.y;
	size_t b_col_base = BN * blockIdx.z;
	size_t batch_idx = blockIdx.x;
	float* A = a.data + batch_idx * a.shape[1] * a.shape[2] + a_row_base * a.shape[2];
	float* B = b.data + batch_idx * b.shape[1] * b.shape[2] + b_col_base;
	float* RES = res.data + batch_idx * res.shape[1] * res.shape[2] + a_row_base * res.shape[2] + b_col_base;

	__shared__ float A_s[BM * BK];
	__shared__ float B_s[BK * BN];

	// store results in registers
	// each thread will be responsible for TM * TN entries of C...
	// TM rows and TN cols, the rows strided every TPBM, the cols strided every TPBN
	float tmp[TM * TN] = {0};

	size_t this_thread_row = threadIdx.x / TPBN;
	size_t this_thread_col = threadIdx.x % TPBN;

	for(size_t k0 = 0; k0 < K; k0 += BK) {
		__syncthreads();
		// load a and b
		// A_s[i, j] = A[a_row + i, k + j]

		// values to fill: BM * BK
		// threads: TPB
		// want: blockDim.x div. by BK
		for(size_t a_idx = threadIdx.x; a_idx < BM * BK; a_idx += blockDim.x) {
			size_t i = a_idx / BK;
			size_t j = a_idx % BK;
			if (k0 + j < K && i + a_row_base < M) {
				A_s[i * BK + j] = A[i * (a.shape[2]) + k0 + j];
			} else {
				A_s[i * BK + j] = 0;
			}
		}

		// B_s[i, j] = B[k + i, b_col + j]
		for(size_t b_idx = threadIdx.x; b_idx < BK * BN; b_idx += blockDim.x) {
			size_t i = b_idx / BN;
			size_t j = b_idx % BN;
			if (k0 + i < K && j + b_col_base < N) {
				B_s[i * BN + j] = B[(k0 + i) * b.shape[2] + j];
			} else {
				B_s[i * BN + j] = 0;
			}
		}

		__syncthreads();
		for(size_t k = 0; k < BK; k++) {
			for(size_t row_counter = 0; row_counter < TM; row_counter ++) {
				// filling in row a_row_base + row_counter * TPBM + this_thread_row of C
				// which is row row_counter of tmp
				size_t row = row_counter * TPBM + this_thread_row;
				float a_val = A_s[row * BK + k];
				for(size_t col_counter = 0; col_counter < TN; col_counter ++) {
					// col b_col_base + col_counter * TPBN + this_thread_col of C
					// which is col col_counter of tmp
					size_t col = col_counter * TPBN + this_thread_col;
					float b_val = B_s[k * BN + col];

					tmp[row_counter * TN + col_counter] += a_val * b_val;
				}
			}
		}
	}
	// now tmp[row * TN + col] goes into RES[(row * TPBM + this_thread_row) * res.shape[2] + (col * TPBN + this_thread_col)]
	for(size_t row_counter = 0; row_counter < TM; row_counter ++) {
		for(size_t col_counter = 0; col_counter < TN; col_counter ++) {
			RES[(row_counter * TPBM + this_thread_row) * res.shape[2] + (col_counter * TPBN + this_thread_col)] = tmp[row_counter * TN + col_counter];
		}
	}
}

__global__ void transpose(ContiguousTensor3d_Device a, ContiguousTensor3d_Device b) {
	// a: (batch, r, c)
	// b: (batch, c, r)
	
	// first: collaboratively copy from a to shared memory
	__shared__ float data[36*36];

	// one warp = one threadIdx.y... want varying x to be the small dimension
	size_t batch = blockIdx.z;
	size_t a_row = 16 * blockIdx.x + threadIdx.y;
	size_t a_col = 16 * blockIdx.y + 4 * threadIdx.x;

	size_t data_row = threadIdx.y;
	size_t data_col = 4 * threadIdx.x;

	size_t a_idx = batch * a.shape[1] * a.shape[2] + a_row * a.shape[2] + a_col;
	size_t data_idx = data_row * 36 + data_col;

	if (a_row < a.shape[1] && a_col < a.shape[2]) {
		reinterpret_cast<float4*>(&data[data_idx])[0] = reinterpret_cast<float4*>(&a.data[a_idx])[0];
	}

	__syncthreads();

	// then: copy from shared memory to b
	size_t b_row = 16 * blockIdx.y + threadIdx.y;
	size_t b_col = 16 * blockIdx.x + 4 * threadIdx.x;

	data_row = 4 * threadIdx.x;
	data_col = threadIdx.y;

	size_t b_idx = batch * b.shape[1] * b.shape[2] + b_row * b.shape[2] + b_col;
	data_idx = data_row * 36 + data_col;

	float4 tmp;
	tmp.x = data[data_row * 36 + data_col];
	tmp.y = data[(data_row + 1) * 36 + data_col];
	tmp.z = data[(data_row + 2) * 36 + data_col];
	tmp.w = data[(data_row + 3) * 36 + data_col];

	if (b_row < b.shape[1] && b_col < b.shape[2]) {
		reinterpret_cast<float4*>(&b.data[b_idx])[0] = tmp;
	}
}

// ============================= Launchers
// ======================================
__host__ void launch_randn(float *a, size_t len, int seed) {
  curandGenerator_t gen;

  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CALL(curandGenerateNormal(gen, a, len, 0.0, 1.0));

  return;
}

__host__ void launch_add_contiguous(float *a, float *b, float *res,
                                    size_t len) {
  if (len < TARGET_THREADS_PER_BLOCK) {
    add_contiguous<<<1, len>>>(a, b, res, len);
  } else {
    size_t n_blocks =
        (len + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    add_contiguous<<<n_blocks, TARGET_THREADS_PER_BLOCK>>>(a, b, res, len);
  }
  return;
}

__host__ int launch_is_eq(FloatTensor *a, FloatTensor *b) {
  int res_h = 1;
  int *res_d;
  CUDA_CHECK(cudaMalloc(&res_d, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(res_d, &res_h, sizeof(int), cudaMemcpyHostToDevice));

  size_t num_entries = a->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  is_eq<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), a->shape_.data(), a->strides_.data(),
      a->offset_, b->strides_.data(), b->offset_, a->dim_, res_d);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&res_h, res_d, sizeof(int), cudaMemcpyDeviceToHost));
  return res_h;
}

__host__ void launch_contiguous_clone(FloatTensor *a, FloatTensor *res) {
	size_t num_entries = a->numel();
	size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  contiguous_clone<<<n_blocks, threads_per_block>>>(
		  a->data_ptr(), res->data_ptr(), a->shape_.data(), a->strides_.data(),
		  a->offset_, a->dim_);
}

__host__ void launch_add(FloatTensor *a, FloatTensor *b, FloatTensor *res) {
  size_t num_entries = a->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  add<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(),
      a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_sub(FloatTensor *a, FloatTensor *b, FloatTensor *res) {
  size_t num_entries = a->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  sub<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(),
      a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_mul(FloatTensor *a, FloatTensor *b, FloatTensor *res) {
  size_t num_entries = a->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  mul<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(),
      a->strides_.data(), a->offset_, b->strides_.data(), b->offset_, a->dim_);
}

__host__ void launch_matmul(FloatTensor *a, FloatTensor *b, FloatTensor *res) {
  size_t num_entries = res->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  matmul<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), res->data_ptr(), a->shape_.data(),
      a->strides_.data(), a->offset_, b->shape_.data(), b->strides_.data(),
      b->offset_, a->dim_);
}

ContiguousTensor3d_Device to_ct3d(FloatTensor *x) {
  return ContiguousTensor3d_Device{x->data_ptr(),
                                   {x->shape_[0], x->shape_[1], x->shape_[2]}};
}

__host__ void launch_matmul_3d(FloatTensor *a, FloatTensor *b,
                               FloatTensor *res) {
  ContiguousTensor3d_Device a_device = to_ct3d(a);
  ContiguousTensor3d_Device b_device = to_ct3d(b);
  ContiguousTensor3d_Device res_device = to_ct3d(res);

  size_t num_entries = res->numel();

  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  matmul_3d<<<n_blocks, threads_per_block>>>(a_device, b_device, res_device);
}

__host__ void launch_matmul_tiled(FloatTensor *a, FloatTensor *b,
                                  FloatTensor *res) {
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
    tpb_z = TPB_ROW;
    grid_dim_z = (row_threads + TPB_ROW - 1) / TPB_ROW;
  } else {
    tpb_z = row_threads;
    grid_dim_z = 1;
  }

  // cols
  size_t col_threads = (n_cols + TILE_COLS - 1) / TILE_COLS;

  if (col_threads > TPB_COL) {
    tpb_y = TPB_COL;
    grid_dim_y = (col_threads + TPB_COL - 1) / TPB_COL;
  } else {
    tpb_y = col_threads;
    grid_dim_y = 1;
  }

  dim3 blocks(grid_dim_x, grid_dim_y, grid_dim_z);
  dim3 threads(1, tpb_y, tpb_z);

  matmul_tiled<<<blocks, threads>>>(a_device, b_device, res_device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void launch_matmul_cublas(FloatTensor *a, FloatTensor *b,
                                   FloatTensor *res) {
  ContiguousTensor3d_Device a_device = to_ct3d(a);
  ContiguousTensor3d_Device b_device = to_ct3d(b);
  ContiguousTensor3d_Device res_device = to_ct3d(res);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // scalars alpha and beta will be stored on host
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  cublasPointerMode_t mode;
  std::cout << "Pointer mode: " << cublasGetPointerMode(handle, &mode) << std::endl;

  // since cublas expects matrices in column-major order
  // we need to feed cublas B-transpose first, then A-transpose
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  int m = (int)b_device.shape[2];
  int n = (int)a_device.shape[1];
  int k = (int)a_device.shape[2];

  int lda = k;
  int ldb = m;
  int ldc = m;

  float alpha = 1.0;
  float beta = 0.0;

  float *A = a_device.data;
  float *B = b_device.data;
  float *C = res_device.data;

  long long int stridea = a_device.shape[1] * a_device.shape[2];
  long long int strideb = b_device.shape[1] * b_device.shape[2];
  long long int stridec = res_device.shape[1] * res_device.shape[2];

  int batch_count = a_device.shape[0];

  std::cout << "Params: m=" << m << ", n=" << n << ", k=" << k << std::endl;
  std::cout << "lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;
  std::cout << "stridea=" << stridea << ", strideb=" << strideb << ", stridec=" << stridec << std::endl;

  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, transb, transa, m, n, k, &alpha, B, ldb, strideb, A, lda, stridea, 
      &beta, C, ldc, stridec, batch_count));

  cublasDestroy(handle);
}

__host__ void launch_matmul_tiled_2(FloatTensor *a, FloatTensor *b, FloatTensor *res) {
	// blocks: (batch, row, col)
	// threads: (n_threads, 1, 1)  -- I will manage it myself
	
  ContiguousTensor3d_Device a_device = to_ct3d(a);
  ContiguousTensor3d_Device b_device = to_ct3d(b);
  ContiguousTensor3d_Device res_device = to_ct3d(res);

  size_t gridDim_x = a_device.shape[0];
  size_t gridDim_y = (a_device.shape[1] + BM - 1) / BM;
  size_t gridDim_z = (b_device.shape[2] + BN - 1) / BN;

  size_t n_threads = TPBM * TPBN;

  dim3 gridDim(gridDim_x, gridDim_y, gridDim_z);
  dim3 blockDim(n_threads, 1, 1);

  matmul_tiled_2<<<gridDim, blockDim>>>(a_device, b_device, res_device);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

}

__host__ void launch_transpose(FloatTensor *a, FloatTensor *b) {
  ContiguousTensor3d_Device a_device = to_ct3d(a);
  ContiguousTensor3d_Device b_device = to_ct3d(b);

  size_t batch = a_device.shape[0];
  size_t a_rows = a_device.shape[1];
  size_t a_cols = a_device.shape[2];

  assert(b_device.shape[0] == a_device.shape[0]);
  assert(b_device.shape[1] == a_device.shape[2]);
  assert(b_device.shape[2] == a_device.shape[1]);

  size_t grid_x = (a_rows + 15) / 16;
  size_t grid_y = (a_cols + 15) / 16;
  size_t grid_z = batch;
  size_t block_x = 4;
  size_t block_y = 16;

  dim3 gridDim(grid_x, grid_y, grid_z);
  dim3 blockDim(block_x, block_y);

  transpose<<<gridDim, blockDim>>>(a_device, b_device);
}
