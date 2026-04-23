#include "tensor.h"

void launch_abs(FloatTensor *in, FloatTensor *out);
void launch_exp(FloatTensor *in, FloatTensor *out);
void launch_log(FloatTensor *in, FloatTensor *out);
void launch_sqrt(FloatTensor *in, FloatTensor *out);
void launch_relu(FloatTensor *in, FloatTensor *out);
void launch_scalar_mul(FloatTensor *in, FloatTensor *out, float c);
void launch_reciprocal(FloatTensor *in, FloatTensor *out);
