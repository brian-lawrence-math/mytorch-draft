
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cassert>

#include "cuda_utils.h"
#include "tensor.h"

void bench_matmul() {
	FloatTensor t1 = FloatTensor::randn({100, 1000, 1000}, Device::GPU);
	FloatTensor t2 = FloatTensor::randn({100, 1000, 1000}, Device::GPU);

	t1.move_to_device(Device::GPU);
	t2.move_to_device(Device::GPU);

	cudaDeviceSynchronize();
	auto start = std::chrono::steady_clock::now();

	FloatTensor res = t1.matmul_tiled(t2);

	cudaDeviceSynchronize();
	auto stop = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed = stop - start;
	std::cout << "Time elapsed: " << elapsed.count() << std::endl;
}

int main() {
	bench_matmul();
	return 0;
}

	
