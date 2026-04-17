# Mytorch: a mock tensor library written in CUDA

I want to rebuild Pytorch for myself, from scratch, on GPU --
and along the way, learn something about performance computing and GPU optimization.

The project features:
- Tensor operations on CPU and GPU
- Optimized batched matrix multiplication for RTX 3050.

What this is not:
- A general-purpose library for production.  It's only configured to run on one particular system.
- A full replacement for Pytorch.  It's missing a lot of features, most notably:
  - Autograd (automatic differentiation)
  - Optimization tools (optimizers and learning rate schedulers).

How to run:
```
$ cmake -S . -B build
$ cmake --build build
$ export PYTHONPATH="src/python"
$ uv run python
>>> from mytorch import FloatTensor as FT, CPU, GPU
>>> x = FT.randn((5,), CPU)
>>> x
Tensor:
  Shape: [5], Offset: 0, Strides: [1]
  Raw data: [1.358244, -0.330076, -0.618854, -1.219036, 0.150899]
```

## Optimization

Along the way I got distracted trying to optimize matrix multiplication.
Read more about that [here](https://brian-lawrence-math.github.io/2026/04/17/opt.html).

## AI Agents?

Somewhat stubbornly I wanted to code this project up by hand.
After all, I'm new to both C++ and CUDA.  I want to learn to write the code myself first,
before I supervise an AI.
I'm looking for the best way to learn, not the fastest way to ship code.
(The one exception is the unit tests, many of which were written by Codex.)

