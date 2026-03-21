#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cassert>

#include "cuda_utils.h"
#include "tensor.h"

void bench_matmul() {
	FloatTensor t1 = FloatTensor::randn({1, 300, 40}, Device::GPU);
	FloatTensor t2 = FloatTensor::randn({1, 40, 20}, Device::GPU);

	t1.move_to_device(Device::GPU);
	t2.move_to_device(Device::GPU);

	FloatTensor res = t1.matmul_tiled(t2);
}

int main() {
	bench_matmul();
	return 0;
}

	
