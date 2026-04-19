#include <cmath>
#include <stdexcept>

#include "cuda_utils.h"
#include "tensor.h"
#include "utils.cuh"

// template time yay
// this will compute any pointwise function on tensor_in

// templates must be implemented in headers, so this whole thing is a header
// file...

template <typename Func>
__global__ void pointwise_kernel(float *tensor_in, float *tensor_out,
                                 size_t *shape, ssize_t *in_strides,
                                 size_t in_offset, ssize_t *out_strides,
                                 size_t out_offset, size_t dim, Func f) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t num_entries = product_device(shape, dim);

  if (idx < num_entries) {
    size_t in_idx =
        flat_idx_to_raw_idx_device(idx, shape, in_strides, in_offset, dim);
    size_t out_idx =
        flat_idx_to_raw_idx_device(idx, shape, out_strides, out_offset, dim);

    tensor_out[out_idx] = f(tensor_in[in_idx]);
  }
}

template <typename Func>
__host__ void launch_pointwise_kernel(FloatTensor *in, FloatTensor *out,
                                      Func f) {
  size_t num_entries = in->numel();
  size_t n_blocks, threads_per_block;
  if (num_entries < TARGET_THREADS_PER_BLOCK) {
    n_blocks = 1;
    threads_per_block = num_entries;
  } else {
    n_blocks =
        (num_entries + TARGET_THREADS_PER_BLOCK - 1) / TARGET_THREADS_PER_BLOCK;
    threads_per_block = TARGET_THREADS_PER_BLOCK;
  }
  pointwise_kernel<<<n_blocks, threads_per_block>>>(
      in->data_ptr(), out->data_ptr(), in->shape_.data(), in->strides_.data(),
      in->offset_, out->strides_.data(), out->offset_, in->dim_, f);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct AbsOp {
  __host__ __device__ float operator()(float x) const { return std::abs(x); }
};

struct ExpOp {
  __host__ __device__ float operator()(float x) const { return expf(x); }
};

struct LogOp {
  __host__ __device__ float operator()(float x) const { return logf(x); }
};

struct SqrtOp {
  __host__ __device__ float operator()(float x) const { return sqrtf(x); }
};

struct ReLUOp {
  __host__ __device__ float operator()(float x) const { return x > 0 ? x : 0; }
};

void launch_abs(FloatTensor *in, FloatTensor *out) {
  launch_pointwise_kernel(in, out, AbsOp{});
}

void launch_exp(FloatTensor *in, FloatTensor *out) {
  launch_pointwise_kernel(in, out, ExpOp{});
}

void launch_log(FloatTensor *in, FloatTensor *out) {
  launch_pointwise_kernel(in, out, LogOp{});
}

void launch_sqrt(FloatTensor *in, FloatTensor *out) {
  launch_pointwise_kernel(in, out, SqrtOp{});
}

void launch_relu(FloatTensor *in, FloatTensor *out) {
  launch_pointwise_kernel(in, out, ReLUOp{});
}
