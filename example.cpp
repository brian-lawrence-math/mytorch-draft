#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

int add(int i, int j){
	return i + j;
}

struct FloatTensor {
	std::vector<float> data;
	size_t dim;
	std::vector<size_t> shape;
	std::vector<size_t> strides;

	FloatTensor(std::vector<float> data, size_t dim, std::vector<size_t> shape, std::vector<size_t> strides) : data(data), dim(dim), shape(shape), strides(strides) {}

	static FloatTensor from_list(std::vector<float> vals) {
		return FloatTensor(vals, 1, {vals.size()}, {1});
	}

	void _validate_1d_index(size_t idx) {
		if (this->dim != 1) {
			throw std::out_of_range("__getitem__ only implemented for 1d tensors");
		}
		if (this->strides[0] != 1) {
			throw std::runtime_error("__getitem__ only implemented for tensors in natural order");
		}
		if (idx < 0 || idx >= this->data.size()) {
			throw std::out_of_range("Index out of range.");
		}
	}

	float _get_from_1d(size_t idx) {
		this->_validate_1d_index(idx);
		return this->data[idx];
	}

	void _set_to_1d(size_t idx, float val) {
		this->_validate_1d_index(idx);
		this->data[idx] = val;
	}
};

PYBIND11_MODULE(example, m, py::mod_gil_not_used()) {
	m.doc() = "is this a docstring?";
	m.def("add", &add, "Add two numbers");
	py::class_<FloatTensor>(m, "FloatTensor")
		.def_static("from_list", &FloatTensor::from_list)
		.def("__getitem__", &FloatTensor::_get_from_1d)
		.def("__setitem__", &FloatTensor::_set_to_1d);
}
