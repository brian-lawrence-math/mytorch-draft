
void launch_randn(float *a, size_t len, int seed);
void launch_add_contiguous(float *a, float *b, float *res, size_t len);
int launch_is_eq(FloatTensor *a, FloatTensor *b);
void launch_add(FloatTensor *a, FloatTensor *b, FloatTensor *res);
void launch_sub(FloatTensor *a, FloatTensor *b, FloatTensor *res);
void launch_mul(FloatTensor *a, FloatTensor *b, FloatTensor *res);
void launch_matmul(FloatTensor *a, FloatTensor *b, FloatTensor *res);
void launch_matmul_3d(FloatTensor *a, FloatTensor *b, FloatTensor *res);
void launch_matmul_tiled(FloatTensor *a, FloatTensor *b, FloatTensor *res);
