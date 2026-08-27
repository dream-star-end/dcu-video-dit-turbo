# gfx936 exact rowwise INT8 quantizer experiment

Research question: can one gfx936 HIP kernel reproduce the deployed serving stack's eager BF16
rowwise INT8 quantization bitwise while saving more than one second over the four
quantization calls in 50 blocks x 20 diffusion steps?

Null hypothesis: either q/scale differ bitwise, or projected full20 saving is at
most one second. Alternative: all correctness cases are bitwise and projected
saving exceeds one second.

Code map: `quantizer_kernel.hip` owns the kernel and PyTorch wrapper;
`quantizer.cpp` owns pybind; `bench_exact_quantizer.py` owns correctness and paired
event timing. The accepted eager implementation remains read-only.

Stop conditions: stop on any bitwise mismatch; otherwise run 11 interleaved timing
rounds with 20 calls per round for M=11819 and K=5376/7168/14336.
