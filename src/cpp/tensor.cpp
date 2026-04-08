#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"
#include "cuda_utils.h"
#include "kernels.cuh"

// ======================== Memory management ========================
float *alloc_float(size_t n, Device dev) {
  void *p;
  switch (dev) {
  case Device::CPU:
    p = ::operator new(n * sizeof(float));
    break;
  case Device::GPU:
    p = nullptr;
    // temp hack to make a little extra space
    n += (16 - (n % 8));
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    break;
  default:
    throw std::runtime_error("Invalid device; code should be unreachable.");
  }
  return static_cast<float *>(p);
}

void dealloc_float(float *p, Device dev) {
  switch (dev) {
  case Device::CPU:
    ::operator delete(p);
    return;
  case Device::GPU:
    CUDA_CHECK(cudaFree(p));
    return;
  }
}

float read_float(float *p, Device dev) {
  switch (dev) {
  case Device::CPU:
    return *p;
  case Device::GPU: {
    float result =
        0.0; // block compiler warning "result may be returned uninitialized"
    CUDA_CHECK(cudaMemcpy(&result, p, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
  }
  default:
    throw std::runtime_error("Invalid device; code should be unreachable.");
  }
}

void write_float(float *p, float val, Device dev) {
  switch (dev) {
  case Device::CPU:
    *p = val;
    break;
  case Device::GPU:
    CUDA_CHECK(cudaMemcpy(p, &val, sizeof(float), cudaMemcpyHostToDevice));
    break;
  }
}

void copy_floats(float *dst, Device dst_dev, float *src, Device src_dev,
                 size_t n) {
  size_t n_bytes = n * sizeof(float);
  switch (dst_dev) {
  case Device::CPU:
    switch (src_dev) {
    case Device::CPU:
      std::memcpy(dst, src, n_bytes);
      break;
    case Device::GPU:
      CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToHost));
      break;
    }
    break;
  case Device::GPU:
    switch (src_dev) {
    case Device::CPU:
      CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyHostToDevice));
      break;
    case Device::GPU:
      CUDA_CHECK(cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToDevice));
      break;
    }
  }
}

template <typename T> std::string array_to_string(T *arr, size_t n) {
  std::string result = "[";
  for (size_t i = 0; i < n; i++) {
    result += std::to_string(*(arr + i));
    if (i < n - 1) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

template <typename T> std::string vector_to_string(std::vector<T> v) {
  return array_to_string(v.data(), v.size());
}

// ========================== Helper methods for indexing =====================

// product of a list of size_t or ssize_t
template <typename T>
T product(std::vector<T> vals) {
  T cml_prod = 1;
  for (size_t val : vals) {
    cml_prod *= val;
  }
  return cml_prod;
}

// Helper method to convert tensor shape to tensor strides (row-major, following
// pytorch)
std::vector<ssize_t> reverse_cml_prod(std::vector<size_t> vals) {
  std::vector<ssize_t> result(vals.size());
  size_t cml_prod = 1;
  for (size_t i = vals.size(); i-- > 0;) {
    result[i] = cml_prod;
    cml_prod *= vals[i];
  }
  return result;
}

LogicalIndex flat_idx_to_idx(const FlatLogicalIndex &idx,
                             std::vector<size_t> shape) {
  size_t flat_idx = idx.idx;
  if (flat_idx > product(shape)) {
    throw std::out_of_range("Index into tensor out of range.");
  }

  std::vector<ssize_t> result(shape.size());
  for (size_t d = shape.size(); d-- > 0;) {
    result[d] = flat_idx % shape[d];
    flat_idx /= shape[d];
  }
  return LogicalIndex{result};
}

// RAII: A FloatBlock manages the pointer to an array of floats.
// The FloatBlock owns the memory: when the FloatBlock is destroyed,
// the memory is deallocated.
// Note that this pointer must be unique: only one FloatBlock should point
// to the same data.
struct FloatBlock {
  // Doing manual memory management, rather than a smart pointer,
  // to build understanding.
  // Note this will require separate alloc/dealloc logic for CPU and GPU.
  float *data;
  size_t size;
  Device dev;

private:
  FloatBlock(size_t size, Device dev) {
    float *data = alloc_float(size, dev);
    this->data = data;
    this->size = size;
    this->dev = dev;
  }

public:
  ~FloatBlock() { dealloc_float(data, dev); }

  // disable copying
  FloatBlock(const FloatBlock &) = delete;
  FloatBlock &operator=(const FloatBlock &) = delete;

  // wrapper around the raw constructor
  // force myself to type out "uninitialized" explicitly if I want this odd
  // behavior
  static FloatBlock *uninitialized(size_t size, Device dev) {
    FloatBlock *block = new FloatBlock(size, dev);
    return block;
  }

  static FloatBlock *zeros_cpu(size_t size) {
    FloatBlock *block = new FloatBlock(size, Device::CPU);
    std::fill(block->data, block->data + size, 0.0f);
    return block;
  }

  static FloatBlock *zeros_gpu(size_t size) {
    FloatBlock *block = new FloatBlock(size, Device::GPU);
    CUDA_CHECK(cudaMemset(block->data, 0, size * sizeof(float)));
    return block;
  }

  static FloatBlock *randn_cpu(size_t size) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d(0, 1);

    FloatBlock *block = new FloatBlock(size, Device::CPU);
    for (size_t idx = 0; idx < size; idx++) {
      float rand_val = d(gen);
      *(block->data + idx) = rand_val;
    }
    return block;
  }

  static FloatBlock *randn_gpu(size_t size) {
    FloatBlock *block = new FloatBlock(size, Device::GPU);

    // generate seed on CPU
    std::random_device rd{};
    int seed = rd();

    launch_randn(block->data, size, seed);
    return block;
  }

  void validate_raw_idx(size_t idx) {
    if (idx < 0 || idx >= size) {
      throw std::out_of_range("Index out of range.");
    }
  }

  // Get the float at index idx of the underlying memory:
  // no shape/stride logic.
  float get_raw_idx(size_t idx) {
    validate_raw_idx(idx);
    return read_float(data + idx, dev);
  }

  void set_raw_idx(size_t idx, float val) {
    validate_raw_idx(idx);
    write_float(data + idx, val, dev);
  }

  FloatBlock *clone(Device new_dev) {
    FloatBlock *new_block = new FloatBlock(size, new_dev);
    copy_floats(new_block->data, new_dev, this->data, this->dev, size);

    return new_block;
  }
};

// =======================================================
// ===================== FloatTensor =====================
// =======================================================

// ===================== Constructors ====================
FloatTensor::FloatTensor(std::shared_ptr<FloatBlock> block, size_t dim,
                         std::vector<size_t> shape, size_t offset,
                         std::vector<ssize_t> strides)
    : block_(block), dim_(dim), shape_(shape), offset_(offset),
      strides_(strides) {}

// since FloatBlock cannot be copied,
// need to explicitly write copy constructors for FloatTensor
FloatTensor::FloatTensor(const FloatTensor &other)
    : block_(other.block_), dim_(other.dim_), shape_(other.shape_),
      offset_(other.offset_), strides_(other.strides_) {}

FloatTensor FloatTensor::zeros(std::vector<size_t> shape, Device dev) {
  FloatBlock *block_raw;
  switch (dev) {
  case Device::CPU:
    block_raw = FloatBlock::zeros_cpu(product(shape));
    break;
  case Device::GPU:
    block_raw = FloatBlock::zeros_gpu(product(shape));
    break;
  }
  std::shared_ptr<FloatBlock> block(block_raw);

  size_t dim = shape.size();
  size_t offset = 0;
  std::vector<ssize_t> strides = reverse_cml_prod(shape);
  return FloatTensor{block, dim, shape, offset, strides};
}

// Wrapper function for convenience
FloatTensor FloatTensor::zeros_1d(size_t s) {
  return FloatTensor::zeros({s}, Device::CPU);
}

FloatTensor FloatTensor::from_list_1d(std::vector<float> vals, Device dev) {
  FloatBlock *block_raw = FloatBlock::uninitialized(vals.size(), dev);
  std::shared_ptr<FloatBlock> block(block_raw);

  size_t dim = 1;
  std::vector<size_t> shape{vals.size()};
  size_t offset = 0;
  std::vector<ssize_t> strides{1};

  // initialize the values
  copy_floats(block->data, dev, vals.data(), Device::CPU, vals.size());
  return FloatTensor{block, dim, shape, offset, strides};
}

FloatTensor FloatTensor::uninitialized(std::vector<size_t> shape, Device dev) {
  FloatBlock *block_raw = FloatBlock::uninitialized(product(shape), dev);
  std::shared_ptr<FloatBlock> block(block_raw);
  size_t dim = shape.size();
  size_t offset = 0;
  std::vector<ssize_t> strides = reverse_cml_prod(shape);
  return FloatTensor{block, dim, shape, offset, strides};
}

FloatTensor FloatTensor::randn(std::vector<size_t> shape, Device dev) {
  FloatBlock *block_raw;
  switch (dev) {
  case Device::CPU:
    block_raw = FloatBlock::randn_cpu(product(shape));
    break;
  case Device::GPU:
    block_raw = FloatBlock::randn_gpu(product(shape));
    break;
  }
  std::shared_ptr<FloatBlock> block(block_raw);

  size_t dim = shape.size();
  size_t offset = 0;
  std::vector<ssize_t> strides = reverse_cml_prod(shape);
  return FloatTensor{block, dim, shape, offset, strides};
}

// ========================== Printing ============================
std::string FloatTensor::raw_repr() {
  // Raw format with three lines.
  // Tensor:
  //   Shape: [...], Offset: ..., Strides: [...],
  //   Raw data: [...]

  std::string shape_repr = vector_to_string<size_t>(this->shape_);
  std::string offset_repr = std::to_string(this->offset_);
  std::string strides_repr = vector_to_string(this->strides_);
  std::string data_repr = "[Data on GPU]";
  if (this->dev_() == Device::CPU) {
    data_repr = array_to_string(this->block_->data, this->block_->size);
  }

  std::string head = "Tensor:\n";
  std::string metadata = "  Shape: " + shape_repr + ", Offset: " + offset_repr +
                         ", Strides: " + strides_repr + "\n";
  std::string data = "  Raw data: " + data_repr + "\n";

  return head + metadata + data;
}

// =============== Indexing and shape management ==================
float *FloatTensor::data_ptr() { return this->block_->data; }

float FloatTensor::get_raw_idx(size_t idx) {
  return this->block_->get_raw_idx(idx);
}

void FloatTensor::set_raw_idx(size_t idx, float val) {
  this->block_->set_raw_idx(idx, val);
}

size_t FloatTensor::numel() { return product(this->shape_); }

LogicalIndex FloatTensor::validate_and_normalize_idx(const LogicalIndex &idx) {
  if (idx.coords.size() != this->dim_) {
    throw std::length_error("Length of index must match dimension of tensor.");
  }
  std::vector<ssize_t> result{};
  for (size_t d = 0; d < this->dim_; d++) {
    if (idx.coords[d] >= (ssize_t)this->shape_[d] ||
        idx.coords[d] < -(ssize_t)this->shape_[d]) {
      throw std::out_of_range("Index into tensor out of range.");
    }
    size_t norm_coord =
        idx.coords[d] >= 0 ? idx.coords[d] : this->shape_[d] + idx.coords[d];
    result.push_back(norm_coord);
  }
  return LogicalIndex{result};
}

size_t FloatTensor::idx_to_raw_idx(LogicalIndex idx) {
  idx = validate_and_normalize_idx(idx);
  ssize_t raw_idx = this->offset_;
  for (size_t d = 0; d < this->dim_; d++) {
    raw_idx += idx.coords[d] * this->strides_[d];
  }
  return (size_t)raw_idx;
}

float FloatTensor::get_idx(LogicalIndex idx) {
  size_t raw_idx = idx_to_raw_idx(idx);
  return get_raw_idx(raw_idx);
}

void FloatTensor::set_idx(LogicalIndex idx, float val) {
  size_t raw_idx = idx_to_raw_idx(idx);
  set_raw_idx(raw_idx, val);
}

// Wrappers around get_idx and set_idx for Python binding
float FloatTensor::py_get_idx(std::vector<ssize_t> coords) {
  LogicalIndex idx = LogicalIndex{coords};
  return get_idx(idx);
}

void FloatTensor::py_set_idx(std::vector<ssize_t> coords, float val) {
  LogicalIndex idx = LogicalIndex{coords};
  set_idx(idx, val);
}

// Two rather hacky functions to enable manual changes to shape and strides,
// mostly for debugging.

// Resets shapes and strides to give a contiguous view of the underlying base
// FloatBlock. If 'this' currently has non-contiguous shape and strides (i.e. is
// a view), they are discarded. New_shape must be compatible with the size of
// the base FloatBlock.
void FloatTensor::base_and_reshape(std::vector<size_t> new_shape) {
  // Compute strides and offsets
  // new_shape will be validated at the end so we only have to compute the
  // product once
  size_t new_dim = new_shape.size();
  size_t new_offset = 0;
  std::vector<ssize_t> new_strides(new_dim);
  size_t cml_prod = 1;
  for (size_t i = new_dim; i-- > 0;) {
    new_strides[i] = cml_prod;
    cml_prod *= new_shape[i];
  }
  // validate
  if ((size_t)cml_prod != this->block_->size) {
    throw std::invalid_argument(
        "New shape must match size of underlying base FloatBlock.");
  }

  // great, all set, now reassign everything
  this->dim_ = new_dim;
  this->shape_ = new_shape;
  this->offset_ = new_offset;
  this->strides_ = new_strides;
}

// Validate shape, offset, and strides against numel.
// Shape, offset, and strides should represent a valid view.
// That is, any logical index within the bounds of shape,
// when converted to a raw index using offset and strides,
// should result in a raw index within the bounds of numel.
void validate_shape_and_strides(size_t numel, std::vector<size_t> shape,
                                size_t offset, std::vector<ssize_t> strides) {
  ssize_t min_idx = offset;
  ssize_t max_idx = offset;

  size_t dim = shape.size();
  if (strides.size() != dim) {
    throw std::length_error("Shape and strides must have the same length.");
  }
  for (size_t idx = 0; idx < dim; idx++) {
    ssize_t biggest_step =
        (shape[idx] - 1) * strides[idx]; // could be positive or negative
    min_idx += std::min((ssize_t)0, biggest_step);
    max_idx += std::max((ssize_t)0, biggest_step);
  }

  if (min_idx < 0 || max_idx >= numel) {
    throw std::range_error(
        "Shape and strides cannot extend outside the base tensor.");
  }
}

void FloatTensor::base_reshape_restride(std::vector<size_t> new_shape,
                                        size_t new_offset,
                                        std::vector<ssize_t> new_strides) {
  validate_shape_and_strides(this->block_->size, new_shape, new_offset,
                             new_strides);

  size_t new_dim = new_shape.size();

  this->dim_ = new_dim;
  this->shape_ = new_shape;
  this->offset_ = new_offset;
  this->strides_ = new_strides;
}

// Create a new FloatTensor with the same underlying block of memory as this
// but specified shape, offset and strides.
// Note that shape and strides of this are ignored.
FloatTensor FloatTensor::view_raw(std::vector<size_t> new_shape,
                                        size_t new_offset,
                                        std::vector<ssize_t> new_strides) {
	validate_shape_and_strides(this->block_->size, new_shape, new_offset,
                             new_strides);

  size_t new_dim = new_shape.size();

  return FloatTensor{this->block_, new_dim, new_shape, new_offset, new_strides};
}

// Assuming this is a contiguous tensor,
// returns a new contiguous tensor of the given shape.
FloatTensor FloatTensor::view(std::vector<ssize_t> new_shape) {
  if (! this->is_contiguous()) {
    throw std::invalid_argument("Can only call view() on a contiguous tensor.  Consider reshape() instead.");
  }
 
  // If one entry of new_shape is -1, replace with the right value
  // And at the same time validate new_shape: there should be at most one -1
  size_t count = 0;
  ssize_t idx = -1;
  for (ssize_t i = 0; i++; i < new_shape.size()) {
	  if (new_shape[i] <= 0) {
		  if (new_shape[i] == -1) {
			  count++;
			  idx = i;
		  } else {
			  throw std::invalid_argument("Cannot assign a shape with negative or zero entries, except for the placeholder -1.");
		  }
	  }
  }
  if (count > 1) {
	  throw std::invalid_argument("Cannot assign a shape with more than one placeholder -1.");
  }

  if (count == 1) {
	  // new_shape includes -1 and all the other dims
	ssize_t product_other_dims = - product(new_shape);
	new_shape[idx] = this->block_->size / product_other_dims;
  }

  if (product(new_shape) != this->block_->size) {
	  throw std::invalid_argument("Invalid shape: product of dimensions of new shape must match size of existing tensor.");
  }

  // ok now new_shape is validated, and if necessary -1 has been replaced with the correct dimension
	
  // convert to size_t
  std::vector<size_t> new_shape_unsigned(new_shape.size());
  for (size_t i = 0; i < new_shape.size(); i++) {
	  new_shape_unsigned[i] = static_cast<size_t>(new_shape[i]);
  }
  size_t new_dim = new_shape.size();
  size_t new_offset = 0;
  std::vector<ssize_t> new_strides = reverse_cml_prod(new_shape_unsigned);

  return FloatTensor{this->block_, new_dim, new_shape_unsigned, new_offset, new_strides};
}


// =================== Memory management ==========================
Device FloatTensor::dev_() { return this->block_->dev; }

FloatTensor FloatTensor::clone() {
  std::shared_ptr<FloatBlock> new_block(this->block_->clone(this->dev_()));
  return FloatTensor(new_block, dim_, shape_, offset_, strides_);
}

void FloatTensor::move_to_device(Device dev) {
  if (dev == this->dev_()) {
    return;
  }

  // copy the memory
  std::shared_ptr<FloatBlock> new_block(this->block_->clone(dev));
  this->block_ = new_block;

  // now the old this->block will lose one reference,
  // and if necessary it will be automatically deallocated.
  // Metadata remains unchanged.
  return;
}

// ==================== Tensor arithmetic on CPU ==================
void FloatTensor::validate_same_shape(const FloatTensor &other) {
  // check the shapes match
  if (this->shape_ != other.shape_) {
    throw std::invalid_argument("Cannot add tensors of different shapes.");
  }
}

std::vector<size_t>
FloatTensor::validate_matmul_shape(const FloatTensor &other) {
  // check that 'this @ other' is a sensible matmul
  // and if so, return the shape of the product

  size_t tot_dim = this->shape_.size();
  if (other.shape_.size() != tot_dim || tot_dim < 2) {
    throw std::invalid_argument("Attempted matmul with invalid shapes.");
  }

  for (int idx = 0; idx < tot_dim - 2; idx++) {
    if (other.shape_[idx] != this->shape_[idx]) {
      throw std::invalid_argument("Attempted matmul with invalid shapes.");
    }
  }

  // this.shape_[-1] should match other.shape_[-2]
  if (this->shape_[tot_dim - 1] != other.shape_[tot_dim - 2]) {
    throw std::invalid_argument("Attempted matmul with invalid shapes.");
  }

  // shape of the product
  std::vector<size_t> result = this->shape_;
  result[tot_dim - 1] = other.shape_[tot_dim - 1];

  return result;
}

bool FloatTensor::is_eq(FloatTensor &other) {
  if (this->shape_ != other.shape_) {
    return false;
  }

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    int result_int = launch_is_eq(this, &other);
    return (result_int != 0);
  } else {
    for (size_t i = 0; i < this->numel(); i++) {
      LogicalIndex log_idx = flat_idx_to_idx(FlatLogicalIndex{i}, this->shape_);
      if (this->get_idx(log_idx) != other.get_idx(log_idx)) {
        return false;
      }
    }
    return true;
  }
}

FloatTensor FloatTensor::add(FloatTensor &other) {
  validate_same_shape(other);

  // allocate result
  FloatTensor result = FloatTensor::uninitialized(this->shape_, this->dev_());

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::cout << "WOW let's test this first kernel!" << std::endl;
    launch_add(this, &other, &result);
  } else {
    // generic CPU code
    // just fill values one by one
    for (size_t i = 0; i < this->numel(); i++) {
      LogicalIndex log_idx = flat_idx_to_idx(FlatLogicalIndex{i}, this->shape_);
      float val = this->get_idx(log_idx) + other.get_idx(log_idx);
      result.set_idx(log_idx, val);
    }
  }
  return result;
}

FloatTensor FloatTensor::sub(FloatTensor &other) {
  validate_same_shape(other);

  // allocate result
  FloatTensor result = FloatTensor::uninitialized(this->shape_, this->dev_());

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::cout << "WOW let's test this first kernel!" << std::endl;
    launch_sub(this, &other, &result);
  } else {
    // generic CPU code
    // just fill values one by one
    for (size_t i = 0; i < this->numel(); i++) {
      LogicalIndex log_idx = flat_idx_to_idx(FlatLogicalIndex{i}, this->shape_);
      float val = this->get_idx(log_idx) - other.get_idx(log_idx);
      result.set_idx(log_idx, val);
    }
  }
  return result;
}

FloatTensor FloatTensor::mul(FloatTensor &other) {
  validate_same_shape(other);

  // allocate result
  FloatTensor result = FloatTensor::uninitialized(this->shape_, this->dev_());

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::cout << "WOW let's test this first kernel!" << std::endl;
    launch_mul(this, &other, &result);
  } else {
    // generic CPU code
    // just fill values one by one
    for (size_t i = 0; i < this->numel(); i++) {
      LogicalIndex log_idx = flat_idx_to_idx(FlatLogicalIndex{i}, this->shape_);
      float val = this->get_idx(log_idx) * other.get_idx(log_idx);
      result.set_idx(log_idx, val);
    }
  }
  return result;
}

FloatTensor FloatTensor::matmul(FloatTensor &other) {
  std::vector<size_t> result_shape = validate_matmul_shape(other);

  // allocate the memory
  FloatTensor result = FloatTensor::uninitialized(result_shape, this->dev_());

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::cout << "CUDA matmul kernel" << std::endl;
    launch_matmul(this, &other, &result);
  } else {
    // generic CPU code
    for (size_t flat_result_idx = 0; flat_result_idx < product(result_shape);
         flat_result_idx++) {
      // set up indices into all three tensors for the loop
      LogicalIndex result_idx =
          flat_idx_to_idx(FlatLogicalIndex{flat_result_idx}, result_shape);
      LogicalIndex this_idx = result_idx;
      LogicalIndex other_idx = result_idx;
      size_t loop_size = this->shape_[this->dim_ - 1];

      float cml_sum = (float)0.0;
      for (int i = 0; i < loop_size; i++) {
        this_idx.coords[this->dim_ - 1] = i;
        other_idx.coords[other.dim_ - 2] = i;
        float this_val = this->get_idx(this_idx);
        float other_val = other.get_idx(other_idx);
        cml_sum += this_val * other_val;
      }
      result.set_idx(result_idx, cml_sum);
    }
  }
  return result;
}

bool FloatTensor::is_contiguous() {
  if (this->offset_ != 0) {
    return false;
  }
  size_t cml_prod = 1;
  for (size_t idx = this->dim_; idx-- > 0;) {
    if (this->strides_[idx] != cml_prod) {
      return false;
    }
    cml_prod *= this->shape_[idx];
  }
  if (this->block_->size != cml_prod) {
    return false;
  }
  return true;
}

FloatTensor FloatTensor::matmul_3d(FloatTensor &other) {
  if (!this->is_contiguous() && !other.is_contiguous()) {
    throw std::invalid_argument(
        "Function matmul_3d() only accepts contiguous tensors.");
  }

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::vector<size_t> result_shape = validate_matmul_shape(other);

    // allocate the memory
    FloatTensor result = FloatTensor::uninitialized(result_shape, this->dev_());

    std::cout << "CUDA matmul_3d kernel" << std::endl;
    launch_matmul_3d(this, &other, &result);

    return result;
  } else {
    return this->matmul(other);
  }
}

FloatTensor FloatTensor::matmul_tiled(FloatTensor &other) {
  if (!this->is_contiguous() && !other.is_contiguous()) {
    throw std::invalid_argument(
        "Function matmul_3d() only accepts contiguous tensors.");
  }

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::vector<size_t> result_shape = validate_matmul_shape(other);

    // allocate the memory
    FloatTensor result = FloatTensor::zeros(result_shape, this->dev_());

    std::cout << "CUDA matmul_tiled kernel" << std::endl;
    launch_matmul_tiled(this, &other, &result);

    return result;
  } else {
    return this->matmul(other);
  }
}

FloatTensor FloatTensor::matmul_cublas(FloatTensor &other) {
  if (!this->is_contiguous() && !other.is_contiguous()) {
    throw std::invalid_argument(
        "Function matmul_3d() only accepts contiguous tensors.");
  }

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::vector<size_t> result_shape = validate_matmul_shape(other);

    // allocate the memory
    FloatTensor result = FloatTensor::zeros(result_shape, this->dev_());

    std::cout << "CUDA matmul_cublas kernel" << std::endl;
    launch_matmul_cublas(this, &other, &result);

    return result;
  } else {
    return this->matmul(other);
  }
}

FloatTensor FloatTensor::matmul_tiled_2(FloatTensor &other) {
  if (!this->is_contiguous() && !other.is_contiguous()) {
    throw std::invalid_argument(
        "Function matmul_3d() only accepts contiguous tensors.");
  }

  if (this->dev_() == Device::GPU && other.dev_() == Device::GPU) {
    std::vector<size_t> result_shape = validate_matmul_shape(other);

    // allocate the memory
    FloatTensor result = FloatTensor::zeros(result_shape, this->dev_());

    std::cout << "CUDA matmul_tiled_2 kernel" << std::endl;
    launch_matmul_tiled_2(this, &other, &result);

    return result;
  } else {
    return this->matmul(other);
  }
}

FloatTensor FloatTensor::transpose() {
	if (!this->is_contiguous() || this->dim_ != 3) {
		throw std::invalid_argument("Function transpose() requires contiguous 3d argument.");
	}
	if (this->dev_() != Device::GPU) {
		throw std::invalid_argument("Function transpose() only implemented on GPU.");
	}

	std::vector<size_t> result_shape{this->shape_[0], this->shape_[2], this->shape_[1]};

	FloatTensor result = FloatTensor::zeros(result_shape, this->dev_());

	std::cout << "CUDA transpose kernel" << std::endl;
	launch_transpose(this, &result);

	return result;
}
	
