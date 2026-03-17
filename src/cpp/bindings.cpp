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
		.def_readonly("dim", &FloatTensor::dim_)
		.def_readonly("shape", &FloatTensor::shape_)
		.def_readonly("offset", &FloatTensor::offset_)
		.def_readonly("strides", &FloatTensor::strides_)
		.def("zeros_1d", &FloatTensor::zeros_1d)
		.def("from_list", &FloatTensor::from_list_1d)
		.def("randn", &FloatTensor::randn)
		.def("__repr__", &FloatTensor::raw_repr)
		.def("get_raw_idx", &FloatTensor::get_raw_idx)
		.def("set_raw_idx", &FloatTensor::set_raw_idx)
		.def("_getitem", &FloatTensor::py_get_idx)
		.def("_setitem", &FloatTensor::py_set_idx)
		.def("base_and_reshape", &FloatTensor::base_and_reshape)
		.def("base_reshape_restride", &FloatTensor::base_reshape_restride)
		.def_property_readonly("device", &FloatTensor::dev_)
		.def("clone", &FloatTensor::clone)
		.def("move_to_device", &FloatTensor::move_to_device)
		.def("add", &FloatTensor::add)
		.def("sub", &FloatTensor::sub)
		.def("mul", &FloatTensor::mul)
		.def("matmul", &FloatTensor::matmul)
		.def("matmul_3d", &FloatTensor::matmul_3d)
		.def("matmul_tiled", &FloatTensor::matmul_tiled);
}
