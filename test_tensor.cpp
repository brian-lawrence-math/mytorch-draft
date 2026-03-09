#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>

#include "cuda_utils.h"
#include "tensor.h"

// Not a good way to compare floats, it doesn't take scale into account...
// but good enough for us right now because all our values are O(1).
#define ASSERT_ALMOST_EQUAL(lhs, rhs) assert(std::abs((lhs) - (rhs)) < 0.0001);

size_t cuda_mem_free() {
	size_t mem_free, mem_total;
	CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
	return mem_free;
}

void test_tensor_constructor() {
	FloatTensor t = FloatTensor::zeros_1d(10);
	float x = t.get_raw_idx(3);
	assert(x == 0.0f);
	return;
}

void test_value_assignment() {
	FloatTensor t = FloatTensor::zeros_1d(10);
	float val = 2.42;
	t.set_raw_idx(3, val);
	float should_be_val = t.get_raw_idx(3);
	assert(should_be_val == val);
	return;
}

void test_move() {
	FloatTensor t = FloatTensor::zeros_1d(2);
	t.set_raw_idx(0, 1.23);
	t.set_raw_idx(1, 4.56);

	t.move_to_device(Device::GPU);
	t.set_raw_idx(0, 2.34);
	float val0 = t.get_raw_idx(0);
	float val1 = t.get_raw_idx(1);
	ASSERT_ALMOST_EQUAL(val0, 2.34);
	ASSERT_ALMOST_EQUAL(val1, 4.56);

	t.move_to_device(Device::CPU);
	float val2 = t.get_raw_idx(0);
	float val3 = t.get_raw_idx(1);
	ASSERT_ALMOST_EQUAL(val2, 2.34);
	ASSERT_ALMOST_EQUAL(val3, 4.56);
	return;
}

void test_memory_not_wasted() {
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	FloatTensor t = FloatTensor::zeros_1d(1000000);  // use lots of memory
	size_t mem_before = cuda_mem_free();
	t.move_to_device(Device::GPU);
	size_t mem_during = cuda_mem_free();
	t.move_to_device(Device::CPU);
	size_t mem_after = cuda_mem_free();
	assert(mem_after > mem_during);
	assert(mem_after == mem_before);
	return;
}

void test_copy_mutability() {
	FloatTensor t1 = FloatTensor::zeros_1d(2);
	FloatTensor t2 = t1;
	t1.set_raw_idx(0, 1.1);
	t2.set_raw_idx(1, 2.2);

	assert(t1.block_ == t2.block_);
	ASSERT_ALMOST_EQUAL(t1.get_raw_idx(0), 1.1);
	ASSERT_ALMOST_EQUAL(t1.get_raw_idx(1), 2.2);
	ASSERT_ALMOST_EQUAL(t2.get_raw_idx(0), 1.1);
	ASSERT_ALMOST_EQUAL(t2.get_raw_idx(1), 2.2);

	return;
}

void test_clone_mutability() {
	FloatTensor t1 = FloatTensor::zeros_1d(2);
	t1.set_raw_idx(0, 1.1);

	FloatTensor t2 = t1.clone();
	t2.set_raw_idx(1, 2.2);

	t1.set_raw_idx(0, 3.3);

	assert(t1.block_ != t2.block_);
	ASSERT_ALMOST_EQUAL(t1.get_raw_idx(0), 3.3);
	ASSERT_ALMOST_EQUAL(t1.get_raw_idx(1), 0.0);
	ASSERT_ALMOST_EQUAL(t2.get_raw_idx(0), 1.1);
	ASSERT_ALMOST_EQUAL(t2.get_raw_idx(1), 2.2);
	return;
}

int main() {
	test_tensor_constructor();
	test_value_assignment();
	test_move();
	test_memory_not_wasted();
	test_copy_mutability();
	test_clone_mutability();

	std::cout << "All tests passed."  << std::endl;
}
