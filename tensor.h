#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <vector>

// TODO: this should not be exposed
struct FloatBlock {
	float get_raw_idx(size_t idx);
	void set_raw_idx(size_t idx, float val);
};

struct FloatTensor {
	FloatBlock* block;
	size_t dim;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	static FloatTensor zeros_1d(size_t size);
};


#endif
