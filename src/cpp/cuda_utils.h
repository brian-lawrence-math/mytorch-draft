#include <cstdio>

#include <cuda_runtime.h>

// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html
// Giving the error result a weird name to avoid namespace collision
#define CUDA_CHECK(expr) do {										\
	cudaError_t __cudaErrResultzzz = expr;										\
	if (__cudaErrResultzzz != cudaSuccess) {									\
		fprintf(													\
				stderr,												\
				"CUDA runtime error: %s:%i:%d = %s\n",				\
				__FILE__,											\
				__LINE__,											\
				__cudaErrResultzzz,												\
				cudaGetErrorString(__cudaErrResultzzz));						\
		fflush(stderr);												\
		throw std::runtime_error("CUDA error.");					\
	}																\
} while (0);													

