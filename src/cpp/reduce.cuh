#include "tensor.h"

void launch_sum(FloatTensor *in, FloatTensor *out, size_t red_dim);
void launch_product(FloatTensor *in, FloatTensor *out, size_t red_dim);
void launch_max(FloatTensor *in, FloatTensor *out, size_t red_dim);
void launch_min(FloatTensor *in, FloatTensor *out, size_t red_dim);
