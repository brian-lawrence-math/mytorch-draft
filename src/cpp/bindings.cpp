#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
      .def("zeros_1d", &FloatTensor::zeros_1d) // can probably delete
      .def("zeros", &FloatTensor::zeros)
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
      .def("view", &FloatTensor::view)
      .def("reshape", &FloatTensor::reshape)
      .def("is_contiguous", &FloatTensor::is_contiguous)
      .def("contiguous", &FloatTensor::contiguous)
      .def("unsqueeze", &FloatTensor::unsqueeze)
      .def("squeeze", &FloatTensor::squeeze)
      .def("permute", &FloatTensor::permute)
      .def("transpose", &FloatTensor::transpose)
      .def("flatten", &FloatTensor::flatten)
      .def("expand", &FloatTensor::expand)
      .def("repeat", &FloatTensor::repeat)
      .def("indexed_view", &FloatTensor::indexed_view)
      .def("move_to_device", &FloatTensor::move_to_device)
      .def("__eq__", &FloatTensor::is_eq)
      .def("view_assign", &FloatTensor::view_assign)
      .def("add", &FloatTensor::add)
      .def("sub", &FloatTensor::sub)
      .def("mul", &FloatTensor::mul)
      .def("matmul", &FloatTensor::matmul)
      .def("abs", &FloatTensor::abs)
      .def("exp", &FloatTensor::exp)
      .def("log", &FloatTensor::log)
      .def("sqrt", &FloatTensor::sqrt)
      .def("relu", &FloatTensor::relu)
      .def("scalar_mul", &FloatTensor::scalar_mul)
	  .def("sum", &FloatTensor::sum)
	  .def("product", &FloatTensor::product)
	  .def("max", &FloatTensor::max)
	  .def("min", &FloatTensor::min);
}
