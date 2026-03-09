#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <memory>
#include <vector>

enum class Device {
	CPU,
	GPU
};

struct FloatBlock;

struct FloatTensor {
	std::shared_ptr<FloatBlock> block;
	size_t dim;
	std::vector<size_t> shape;
	std::vector<size_t> strides;


	FloatTensor(std::shared_ptr<FloatBlock> block, size_t dim, std::vector<size_t> shape, std::vector<size_t> strides);
	FloatTensor(const FloatTensor& other);
	static FloatTensor zeros_1d(size_t size);

	float get_raw_idx(size_t idx);

	void set_raw_idx(size_t idx, float val);

	Device dev();
};


#endif
