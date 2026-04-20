#include <cassert>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cublas_v2.h"
#include <curand.h>

#include "cuda_utils.h"
#include "pointwise.cuh"
#include "tensor.h"
#include "utils.cuh"

// Hard-coded constant specific to the RTX 3050
#define MAX_THREADS_PER_BLOCK (1024)

// shared mem limits:
//   12288 floats / block
//   25600 floats / SM
//   we only have one block per SM right now anyway

#define TPB_ROW (32)
#define TPB_COL (32)
#define TILE_ROWS (4)
#define TILE_COLS (4)

#define TM (8)
#define TN (8)
#define TPBM (16)
#define TPBN (16)
#define BM (TM * TPBM)
#define BN (TN * TPBN)
#define BK (16)

// make sure vectorized loads are ok
static_assert((BK % 4 == 0) && (BN % 4 == 0));

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
                                 size_t *shape, ssize_t *a_strides,
                                 size_t a_offset, size_t dim) {
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

__global__ void view_assign(float *tensor_a, float *tensor_b, size_t *shape,
                            ssize_t *a_strides, size_t a_offset,
                            ssize_t *b_strides, size_t b_offset, size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t a_idx =
        flat_idx_to_raw_idx_device(idx, shape, a_strides, a_offset, dim);
    size_t b_idx =
        flat_idx_to_raw_idx_device(idx, shape, b_strides, b_offset, dim);

    tensor_a[a_idx] = tensor_b[b_idx];
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

__global__ void matmul_tiled_2(ContiguousTensor3d_Device a,
                               ContiguousTensor3d_Device b,
                               ContiguousTensor3d_Device res) {
  // blocks: (batch, row, col)
  // threads: (n_threads, 1, 1)  -- I will manage indexing myself

  size_t M = a.shape[1], K = a.shape[2], N = b.shape[2];
  size_t a_row_base = BM * blockIdx.y;
  size_t b_col_base = BN * blockIdx.z;
  size_t batch_idx = blockIdx.x;
  float *A =
      a.data + batch_idx * a.shape[1] * a.shape[2] + a_row_base * a.shape[2];
  float *B = b.data + batch_idx * b.shape[1] * b.shape[2] + b_col_base;
  float *RES = res.data + batch_idx * res.shape[1] * res.shape[2] +
               a_row_base * res.shape[2] + b_col_base;

  __shared__ float A_s[BM * BK];
  __shared__ float B_s[BK * BN];

  // store results in registers
  // each thread will be responsible for TM * TN entries of C...
  // TM rows and TN cols, the rows strided every TPBM, the cols strided every
  // TPBN
  float tmp[TM * TN] = {0};

  size_t this_thread_row = threadIdx.x / TPBN;
  size_t this_thread_col = threadIdx.x % TPBN;

  for (size_t k0 = 0; k0 < K; k0 += BK) {
    __syncthreads();
    // load a and b
    // A_s[i, j] = A[a_row + i, k + j]

    // values to fill: BM * BK
    // threads: TPB
    // want: blockDim.x div. by BK
    for (size_t a_idx = 4 * threadIdx.x; a_idx < BM * BK; a_idx += 4 * blockDim.x) {
      size_t i = a_idx / BK;
      size_t j = a_idx % BK;
      if (k0 + j < K && i + a_row_base < M) {
        *reinterpret_cast<float4*>(&A_s[i * BK + j]) = *reinterpret_cast<float4*>(&A[i * (a.shape[2]) + k0 + j]);
      } else {
        A_s[i * BK + j] = 0;
		A_s[i * BK + j + 1] = 0;
		A_s[i * BK + j + 2] = 0;
		A_s[i * BK + j + 3] = 0;
      }
    }

    // B_s[i, j] = B[k + i, b_col + j]
    for (size_t b_idx = threadIdx.x; b_idx < BK * BN; b_idx += blockDim.x) {
      size_t i = b_idx / BN;
      size_t j = b_idx % BN;
      if (k0 + i < K && j + b_col_base < N) {
        B_s[i * BN + j] = B[(k0 + i) * b.shape[2] + j];
      } else {
        B_s[i * BN + j] = 0;
      }
    }

    __syncthreads();
    for (size_t k = 0; k < BK; k++) {
	  float B_reg[TN];

	  for (size_t col_counter = 0; col_counter < TN; col_counter++) {
          size_t col = col_counter * TPBN + this_thread_col;
		  B_reg[col_counter] = B_s[k * BN + col];
	  }

      for (size_t row_counter = 0; row_counter < TM; row_counter++) {
        // filling in row a_row_base + row_counter * TPBM + this_thread_row of C
        // which is row row_counter of tmp
        size_t row = row_counter * TPBM + this_thread_row;
        float a_val = A_s[row * BK + k];
        for (size_t col_counter = 0; col_counter < TN; col_counter++) {
          // col b_col_base + col_counter * TPBN + this_thread_col of C
          // which is col col_counter of tmp
		  float b_val = B_reg[col_counter];

          tmp[row_counter * TN + col_counter] += a_val * b_val;
        }
      }
    }
  }
  // now tmp[row * TN + col] goes into RES[(row * TPBM + this_thread_row) *
  // res.shape[2] + (col * TPBN + this_thread_col)]
  for (size_t row_counter = 0; row_counter < TM; row_counter++) {
    for (size_t col_counter = 0; col_counter < TN; col_counter++) {
      RES[(row_counter * TPBM + this_thread_row) * res.shape[2] +
          (col_counter * TPBN + this_thread_col)] =
          tmp[row_counter * TN + col_counter];
    }
  }
}

__global__ void transpose(ContiguousTensor3d_Device a,
                          ContiguousTensor3d_Device b) {
  // a: (batch, r, c)
  // b: (batch, c, r)

  // first: collaboratively copy from a to shared memory
  __shared__ float data[36 * 36];

  // one warp = one threadIdx.y... want varying x to be the small dimension
  size_t batch = blockIdx.z;
  size_t a_row = 16 * blockIdx.x + threadIdx.y;
  size_t a_col = 16 * blockIdx.y + 4 * threadIdx.x;

  size_t data_row = threadIdx.y;
  size_t data_col = 4 * threadIdx.x;

  size_t a_idx = batch * a.shape[1] * a.shape[2] + a_row * a.shape[2] + a_col;
  size_t data_idx = data_row * 36 + data_col;

  if (a_row < a.shape[1] && a_col < a.shape[2]) {
    reinterpret_cast<float4 *>(&data[data_idx])[0] =
        reinterpret_cast<float4 *>(&a.data[a_idx])[0];
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
    reinterpret_cast<float4 *>(&b.data[b_idx])[0] = tmp;
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

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void launch_view_assign(FloatTensor *a, FloatTensor *b) {
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
  view_assign<<<n_blocks, threads_per_block>>>(
      a->data_ptr(), b->data_ptr(), a->shape_.data(), a->strides_.data(),
      a->offset_, b->strides_.data(), b->offset_, a->dim_);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
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

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
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

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
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

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

ContiguousTensor3d_Device to_ct3d(FloatTensor *x) {
  return ContiguousTensor3d_Device{x->data_ptr(),
                                   {x->shape_[0], x->shape_[1], x->shape_[2]}};
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
  std::cout << "Pointer mode: " << cublasGetPointerMode(handle, &mode)
            << std::endl;

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
  std::cout << "stridea=" << stridea << ", strideb=" << strideb
            << ", stridec=" << stridec << std::endl;

  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, transb, transa, m, n, k, &alpha, B, ldb, strideb, A, lda, stridea,
      &beta, C, ldc, stridec, batch_count));

  cublasDestroy(handle);
}

__host__ void launch_matmul_tiled_2(FloatTensor *a, FloatTensor *b,
                                    FloatTensor *res) {
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
