#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <memory>
#include <vector>

enum class Device { CPU, GPU };

struct FloatBlock;

struct FlatLogicalIndex {
  size_t idx;
};

struct LogicalIndex {
  std::vector<ssize_t> coords;
};

struct FloatTensor {
  std::shared_ptr<FloatBlock> block_;
  size_t dim_;
  std::vector<size_t> shape_;
  size_t offset_;
  std::vector<ssize_t> strides_;

  FloatTensor(std::shared_ptr<FloatBlock> block, size_t dim,
              std::vector<size_t> shape, size_t offset,
              std::vector<ssize_t> strides);
  FloatTensor(const FloatTensor &other);
  static FloatTensor zeros(std::vector<size_t> shape, Device dev);
  static FloatTensor zeros_1d(size_t size);
  static FloatTensor from_list_1d(std::vector<float> vals, Device dev);
  static FloatTensor uninitialized(std::vector<size_t> shape, Device dev);
  static FloatTensor randn(std::vector<size_t> shape, Device dev);
  std::string raw_repr();

  float *data_ptr();
  float get_raw_idx(size_t idx);
  void set_raw_idx(size_t idx, float val);
  size_t numel();
  LogicalIndex validate_and_normalize_idx(const LogicalIndex &idx);
  size_t idx_to_raw_idx(LogicalIndex idx);
  float get_idx(LogicalIndex idx);
  void set_idx(LogicalIndex idx, float val);
  float py_get_idx(std::vector<ssize_t> coords);
  void py_set_idx(std::vector<ssize_t> coords, float val);
  void base_and_reshape(std::vector<size_t> new_shape);
  void base_reshape_restride(std::vector<size_t> new_shape, size_t new_offset,
                             std::vector<ssize_t> new_strides);
  FloatTensor view_raw(std::vector<size_t> new_shape, size_t new_offset,
                       std::vector<ssize_t> new_strides);
  std::vector<size_t> validate_new_shape(std::vector<ssize_t> new_shape);
  FloatTensor view(std::vector<ssize_t> new_shape);
  FloatTensor reshape(std::vector<ssize_t> new_shape);
  FloatTensor indexed_view(std::vector<bool> singleton,
                           std::vector<size_t> shape,
                           std::vector<size_t> rel_offsets,
                           std::vector<ssize_t> rel_strides);
  FloatTensor contiguous_clone();
  FloatTensor unsqueeze(ssize_t new_idx);
  FloatTensor squeeze(ssize_t idx);
  FloatTensor permute(std::vector<ssize_t> dims);
  FloatTensor transpose(ssize_t i, ssize_t j);
  FloatTensor flatten(ssize_t start_dim, ssize_t end_dim);
  FloatTensor contiguous();
  FloatTensor expand(std::vector<ssize_t> new_shape);
  FloatTensor repeat(std::vector<size_t> n_repeats);

  Device dev_();
  FloatTensor clone();
  void move_to_device(Device);

  void validate_same_shape(const FloatTensor &other);
  std::vector<size_t> validate_matmul_shape(const FloatTensor &other);
  bool is_eq(FloatTensor &other);
  void view_assign(FloatTensor &other);
  FloatTensor add(FloatTensor &other);
  FloatTensor sub(FloatTensor &other);
  FloatTensor mul(FloatTensor &other);
  FloatTensor matmul(FloatTensor &other);

  bool is_contiguous();
  FloatTensor matmul_3d(FloatTensor &other);
  FloatTensor matmul_tiled(FloatTensor &other);
  FloatTensor matmul_cublas(FloatTensor &other);
  FloatTensor matmul_tiled_2(FloatTensor &other);

  FloatTensor transpose_special_case();

  template <typename Func>
  FloatTensor pointwise_op(FloatTensor &result, Func f);
  FloatTensor abs();
  FloatTensor exp();
  FloatTensor log();
  FloatTensor sqrt();
  FloatTensor relu();



  template <typename Func>
  FloatTensor reduce_op(FloatTensor &result, size_t red_dim, Func f);

  FloatTensor sum(ssize_t red_dim);
};

#endif
