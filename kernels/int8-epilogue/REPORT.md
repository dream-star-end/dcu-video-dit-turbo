# H3 gfx936 INT8 epilogue experiment

## Outcome

A scratch HIP kernel fusing the current eager INT32 dequantization epilogue
(`accumulator`, FP32 row scale, FP32 column scale, optional bias, BF16 output)
compiled for gfx936 and matched the eager result bitwise in all 15 full-shape
correctness cases.

It is a standalone upper-bound candidate. The kernel was not installed into the
runtime and this sub-experiment did not generate a complete video.

## Contract and build

The kernel preserves the current eager operation order:

1. `combined_scale = row_scale[row] * col_scale[col]` in FP32;
2. `scaled = float(accumulator) * combined_scale` in FP32;
3. round to BF16;
4. if present, round bias to BF16, add in widened FP32, and round again to BF16.

Build target and flags:

```text
--offload-arch=gfx936
-O3 -fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off
```

Artifact:

`kernels/int8-epilogue/build/h3_hip_epilogue.so`

## Correctness

The tested hot H3 matrix height is `M=8216`. The INT8 GEMM may internally pad
to 8224, but the current eager path slices the accumulator back to 8216 before
the epilogue, so 8216 is the correct standalone contract.

Across qkv, out/fc2, and fc1 output widths:

- no bias: three seeds per shape;
- BF16 bias: one seed per shape;
- FP32 bias: one seed per shape;
- total: 15 cases;
- all cases: `torch.equal=True`, `max_abs=0`, `mismatch_count=0`.

Raw result:

`kernels/int8-epilogue/bench_results_m8216_2d.json`

## Timing and allocation

No-bias HIP event medians:

| Shape | Eager | Fused | Speedup | Saved |
|---|---:|---:|---:|---:|
| qkv `[8216,21504]` | 6.016 ms | 2.312 ms | 2.60x | 3.704 ms |
| out/fc2 `[8216,5376]` | 1.418 ms | 0.580 ms | 2.45x | 0.838 ms |
| fc1 `[8216,28672]` | 7.998 ms | 3.081 ms | 2.60x | 4.917 ms |

Peak incremental allocated bytes:

| Shape | Eager | Fused |
|---|---:|---:|
| qkv | 1,243,512,832 | 353,353,728 |
| out/fc2 | 706,707,456 | 88,338,432 |
| fc1 | 1,354,563,584 | 471,203,840 |

For one block (`qkv + out + fc1 + fc2`) the isolated saved time is about
10.297 ms. At 50 blocks and 20 sampler steps, the strict standalone upper bound
is about 10.30 seconds, or 8.28% of the 124.4-second sampler baseline. This is
not an end-to-end proof; integration, allocator effects, and neighboring kernels
can reduce the realized gain.

## Artifacts

- `epilogue.cpp`
- `epilogue_kernel.hip`
- `build_extension.py`
- `bench.py`
- `build_2d.log`
- `bench_results_m8216_2d.json`

