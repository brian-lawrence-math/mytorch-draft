#include <cstdio>

#include <cuda_runtime.h>

// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
// Giving the error result a weird name to avoid namespace collision
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t __cudaErrResultzzz = expr;                                     \
    if (__cudaErrResultzzz != cudaSuccess) {                                   \
      fprintf(stderr, "CUDA runtime error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, __cudaErrResultzzz,                                    \
              cudaGetErrorString(__cudaErrResultzzz));                         \
      fflush(stderr);                                                          \
      throw std::runtime_error("CUDA error.");                                 \
    }                                                                          \
  } while (0);

#define CURAND_CALL(x)                                                         \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);                 \
      throw std::runtime_error("CUDA error.");                                 \
    }                                                                          \
  } while (0)

#define CUDA_STATUS(expr)                                                      \
  do {                                                                         \
    cudaStatus_t __statusResult = expr;                                        \
    if (__statusResult != CUDA_STATUS_SUCCESS) {                               \
      fprintf(stderr, "CUDA runtime error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, __statusResult, cudaGetErrorString(__statusResult));   \
      fflush(stderr);                                                          \
      throw std::runtime_error("CUDA error.");                                 \
    }                                                                          \
  } while (0);

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t __statusResult = expr;                                      \
    if (__statusResult != CUBLAS_STATUS_SUCCESS) {                             \
      fprintf(stderr, "CUDA runtime error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, __statusResult,                                        \
              cublasGetStatusString(__statusResult));                          \
      fflush(stderr);                                                          \
      throw std::runtime_error("CUDA error.");                                 \
    }                                                                          \
  } while (0);
