#include <cstdio>
#include <iostream>
#include <cassert>
#include "tensor.h"

void test_tensor_constructor() {
	FloatTensor t = FloatTensor::zeros_1d(10);
	std::cout << "Tensor constructed" << std::endl;
	float x = t.get_raw_idx(3);
	std::cout << "Value of zero: " << x << std::endl;
	assert(x == 0.0f);
	return;
}

int main() {
	test_tensor_constructor();
}
