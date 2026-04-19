#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cassert>

#include "cuda_utils.h"
#include "tensor.h"

void bench_matmul() {
	FloatTensor t1 = FloatTensor::zeros({1, 4, 6}, Device::GPU);
	FloatTensor t2 = FloatTensor::zeros({1, 6, 2}, Device::GPU);

	t1.set_raw_idx(0, 1.0);
	t2.set_raw_idx(0, 1.0);
	t2.set_raw_idx(1, 2.1);

	t1.move_to_device(Device::GPU);
	t2.move_to_device(Device::GPU);

	FloatTensor res = t1.matmul_cublas(t2);
	res.move_to_device(Device::CPU);
	std::cout << res.raw_repr() << std::endl;

	res = t1.matmul_tiled_2(t2);
	res.move_to_device(Device::CPU);
	std::cout << res.raw_repr() << std::endl;
	
}

int main() {
	bench_matmul();
	return 0;
}

	
