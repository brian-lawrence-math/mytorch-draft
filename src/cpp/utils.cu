
// ========================== Helper methods for indexing =====================
// Modified from tensor.cpp to run on device (i.e. no vectors)
__device__ size_t product_device(size_t *vals, size_t n) {
  size_t cml_prod = 1;
  for (size_t i = 0; i < n; i++) {
    cml_prod *= vals[i];
  }
  return cml_prod;
}

// shape is an array size_t[dim]
// strides is another array size_t[dim] of the same size
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t *shape,
                                             ssize_t *strides, size_t offset,
                                             size_t dim) {
  ssize_t result = offset;
  for (size_t d = dim; d-- > 0;) {
    result += (flat_idx % shape[d]) * strides[d];
    flat_idx /= shape[d];
  }
  return (size_t)result;
}
