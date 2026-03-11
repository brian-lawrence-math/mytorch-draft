#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/native_enum.h>

#include "cuda_utils.h"
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, mytorch, py::mod_gil_not_used()) {
	mytorch.doc() = "A mock of pytorch's tensor library.";

	py::native_enum<Device>(mytorch, "Device", "enum.Enum")
		.value("CPU", Device::CPU)
		.value("GPU", Device::GPU)
		.export_values()
		.finalize();

	py::class_<FloatTensor>(mytorch, "FloatTensor")
		.def("zeros_1d", &FloatTensor::zeros_1d)
		.def("from_list", &FloatTensor::from_list_1d)
		.def("get_raw_idx", &FloatTensor::get_raw_idx)
		.def("set_raw_idx", &FloatTensor::set_raw_idx)
		.def("__getitem__", &FloatTensor::py_get_idx)
		.def("__setitem__", &FloatTensor::py_set_idx)
		.def("base_and_reshape", &FloatTensor::base_and_reshape)
		.def("clone", &FloatTensor::clone)
		.def("move_to_device", &FloatTensor::move_to_device)
		.def("add", &FloatTensor::add)
		.def("sub", &FloatTensor::sub)
		.def("mul", &FloatTensor::mul);
}
