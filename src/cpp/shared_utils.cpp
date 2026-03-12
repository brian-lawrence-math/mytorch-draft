#include <vector>

#include "tensor.h"


// ========================== Helper methods for indexing =====================
size_t product(std::vector<size_t> vals) {
	size_t cml_prod = 1;
	for(size_t val : vals) {
		cml_prod *= val;
	}
	return cml_prod;
}

// Helper method to convert tensor shape to tensor strides (row-major, following pytorch)
std::vector<size_t> reverse_cml_prod(std::vector<size_t> vals) {
	std::vector<size_t> result(vals.size());
	size_t cml_prod = 1;
	for(size_t i = vals.size(); i-- > 0; ) {
		result[i] = cml_prod;
		cml_prod *= vals[i];
	}
	return result;
}

LogicalIndex flat_idx_to_idx(const FlatLogicalIndex& idx, std::vector<size_t> shape) {
	size_t flat_idx = idx.idx;
	if (flat_idx > product(shape)) {
		throw std::out_of_range("Index into tensor out of range.");
	}
	
	std::vector<ssize_t> result(shape.size());
	for(size_t d = shape.size(); d-- > 0; ) {
		result[d] = flat_idx % shape[d];
		flat_idx /= shape[d];
	}
	return LogicalIndex{result};
}
