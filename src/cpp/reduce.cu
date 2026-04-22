#include <cmath>
#include <limits>
#include <stdexcept>

#include "cuda_utils.h"
#include "tensor.h"
#include "utils.cuh"

// template for "reduce" function on tensors 
// i.e. binary associative functions that reduce away one dimension at a time
// e.g. sum, max

// this function assumes that the last dimension is the dimension to be reduced
template <typename ReduceOp>
__global__ void reduce_kernel(float *tensor_in, float *tensor_out,
		size_t *in_shape, ssize_t *in_strides, size_t in_offset,
		size_t *out_shape, ssize_t *out_strides, size_t out_offset, size_t in_dim,
		size_t red_dim, ReduceOp f) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t out_dim = in_dim - 1;

  size_t num_entries_out = product_device(out_shape, out_dim);
  ssize_t reduction_shape = in_shape[red_dim];
  ssize_t reduction_stride = in_strides[red_dim];

  if (idx < num_entries_out) {
    size_t out_idx =
        flat_idx_to_raw_idx_device(idx, out_shape, out_strides, out_offset, out_dim);
    size_t in_idx =
        flat_idx_to_raw_idx_skip_dim_device(idx, in_shape, in_strides, in_offset, in_dim, red_dim);

	float result = f.initial_value();
	for (size_t reduce_idx = 0; reduce_idx < reduction_shape; reduce_idx ++) {
		result = f(result, tensor_in[in_idx]);
		in_idx += reduction_stride;
	}

    tensor_out[out_idx] = result;
  }
}

template <typename ReduceOp>
__host__ void launch_reduce_kernel(FloatTensor *in, FloatTensor *out,
                                      size_t red_dim, ReduceOp f) {
  size_t num_entries_out = out->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries_out < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries_out;
  } else {
    n_blocks =
        (num_entries_out + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  reduce_kernel<<<n_blocks, threads_per_block>>>(
      in->data_ptr(), out->data_ptr(), in->shape_.data(), in->strides_.data(),
      in->offset_, out->shape_.data(), out->strides_.data(), out->offset_, in->dim_, 
	  red_dim, f);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


struct SumOp {
	__host__ __device__ float initial_value() { return 0.0f; }
	__host__ __device__ float operator()(float acc, float val) const {return acc + val;}
};

struct ProductOp {
	__host__ __device__ float initial_value() { return 1.0f; }
	__host__ __device__ float operator()(float acc, float val) const {return acc * val;}
};

struct MaxOp {
	__host__ __device__ float initial_value() { return -INFINITY; }
	__host__ __device__ float operator()(float acc, float val) const {return (acc > val) ? acc : val;}
};

struct MinOp {
	__host__ __device__ float initial_value() { return INFINITY; }
	__host__ __device__ float operator()(float acc, float val) const {return (acc < val) ? acc : val;}
};

void launch_sum(FloatTensor *in, FloatTensor *out, size_t red_dim) {
  launch_reduce_kernel(in, out, red_dim, SumOp{});
}

void launch_product(FloatTensor *in, FloatTensor *out, size_t red_dim) {
  launch_reduce_kernel(in, out, red_dim, ProductOp{});
}

void launch_max(FloatTensor *in, FloatTensor *out, size_t red_dim) {
  launch_reduce_kernel(in, out, red_dim, MaxOp{});
}

void launch_min(FloatTensor *in, FloatTensor *out, size_t red_dim) {
  launch_reduce_kernel(in, out, red_dim, MinOp{});
}

