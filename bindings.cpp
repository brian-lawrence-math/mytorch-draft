#include <pybind11/pybind11.h>

#include "cuda_utils.h"
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(mytorch, mytorch, py::mod_gil_not_used()) {
	mytorch.doc() = "A mock of pytorch's tensor library.";

	py::class_<FloatTensor>(mytorch, "FloatTensor")
		.def("zeros_1d", &FloatTensor::zeros_1d)
		.def("get_raw_idx", &FloatTensor::get_raw_idx)
		.def("set_raw_idx", &FloatTensor::set_raw_idx);
}
