
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

	size_t raw_idx = 14141732;

	t1.move_to_device(Device::GPU);
	t2.move_to_device(Device::GPU);

	cudaDeviceSynchronize();
	auto start = std::chrono::steady_clock::now();

	FloatTensor res = t1.matmul_tiled_2(t2);

	cudaDeviceSynchronize();
	auto stop = std::chrono::steady_clock::now();

	std::cout << "Value: " << res.get_raw_idx(raw_idx) << std::endl;
	std::chrono::duration<double> elapsed = stop - start;
	std::cout << "Time elapsed (tiled_2): " << elapsed.count() << std::endl;


	//t1 = FloatTensor::randn({100, 1000, 1000}, Device::GPU);
	//t2 = FloatTensor::randn({100, 1000, 1000}, Device::GPU);

	//t1.move_to_device(Device::GPU);
	//t2.move_to_device(Device::GPU);

	cudaDeviceSynchronize();
	start = std::chrono::steady_clock::now();

	res = t1.matmul_cublas(t2);

	cudaDeviceSynchronize();
	stop = std::chrono::steady_clock::now();

	std::cout << "Value: " << res.get_raw_idx(raw_idx) << std::endl;
	elapsed = stop - start;
	std::cout << "Time elapsed (cublas): " << elapsed.count() << std::endl;

}

void bench_clone() {
	FloatTensor t1 = FloatTensor::randn({250, 1024, 1024}, Device::GPU);
	cudaDeviceSynchronize();

	auto start = std::chrono::steady_clock::now();

	FloatTensor t2 = t1.clone();
	cudaDeviceSynchronize();

	auto stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = stop - start;
	std::cout << "Time elapsed in clone: " << elapsed.count() << std::endl;

}

void bench_transpose() {
	FloatTensor t1 = FloatTensor::randn({250, 1000, 1000}, Device::GPU);
	cudaDeviceSynchronize();

	auto start = std::chrono::steady_clock::now();
	FloatTensor t2 = t1.transpose_special_case();
	cudaDeviceSynchronize();
	auto stop = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = stop - start;
	std::cout << "Time elapsed in transpose: " << elapsed.count() << std::endl;

	float v1 = t1.get_idx(LogicalIndex{std::vector<ssize_t>{3, 5, 7}});
	float v2 = t2.get_idx(LogicalIndex{std::vector<ssize_t>{3, 7, 5}});
	std::cout << "Values: " << v1 << "=" << v2 << std::endl;

	v1 = t1.get_idx(LogicalIndex{std::vector<ssize_t>{3, 18, 16}});
	v2 = t2.get_idx(LogicalIndex{std::vector<ssize_t>{3, 16, 18}});
	std::cout << "Values: " << v1 << "=" << v2 << std::endl;
}

int main() {
	bench_matmul();
	return 0;
}

	
