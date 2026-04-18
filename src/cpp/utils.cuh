
__device__ size_t product_device(size_t *vals, size_t n);
__device__ size_t flat_idx_to_raw_idx_device(size_t flat_idx, size_t *shape,
                                             ssize_t *strides, size_t offset,
                                             size_t dim);
