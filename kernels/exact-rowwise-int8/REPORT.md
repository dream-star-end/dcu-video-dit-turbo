# Exact gfx936 rowwise INT8 quantizer result

## Outcome

GO for a controlled launcher A/B. The independent DSO passed bitwise q and
FP32-scale gates and projects 6.755 seconds saved over 50 blocks x 20 steps.
The running service and installed comfy-kitchen package were not modified.

## Correctness

`run_full.log` contains 12 full-size comparisons at M=11819: random, all-zero,
finite BF16 extremes, and exact round-even boundaries for K=5376, 7168, and
14336. Every q mismatch count and scale-bit mismatch count is zero.

`robustness.log` adds two random seeds for each K, non-unit scale=0.5
round-even boundaries for each K, and a NaN/+Inf/-Inf case. All ten additional
comparisons also have zero q and scale-bit mismatches.

## Performance

Each number is the median of 11 alternating-order HIP-event rounds with 20 calls
per round after 12 warmups on one BW/gfx936 card.

| M x K | eager ms | DSO ms | speedup | saved ms/call |
|---|---:|---:|---:|---:|
| 11819 x 5376 | 1.682118 | 0.526248 | 3.196x | 1.155871 |
| 11819 x 7168 | 2.203438 | 0.695840 | 3.167x | 1.507599 |
| 11819 x 14336 | 4.305621 | 1.370383 | 3.142x | 2.935238 |

Production weighting is K5376 x2000, K7168 x1000, K14336 x1000, yielding a
projected 6754.58 ms (6.755 s) full20 saving. This is a component-level
projection; the controlled launcher A/B is still required to measure realized
end-to-end gain.

## Candidate integration

`benchmark_launcher_exact_quant.py` is based on the read-only
`h3_recovered_runtime/benchmark_launcher_v2.py`. It loads the DSO at startup and
hard-fails if missing/incompatible, then gates use to M=11819, K in
{5376,7168,14336}, contiguous BF16, gfx936, and stochastic_rounding=0. All other
inputs use the original eager quantizer. First-hit, every-500-hit, first fallback
reason, and final counters are logged. `benchmark_launcher_exact_quant.patch`
is the corresponding unified diff.

`launcher_gate_test.log` records an isolated import/gate test without starting
ComfyUI: one eligible M=11819,K=5376 tensor produced one DSO hit, while an M=7
tensor used eager and recorded fallback reason `M`. Startup DSO loading succeeded.

## Evaluation summary

- evaluation_summary: exactness gate passed on 22 comparisons; performance gate passed by 6.75x the required saving.
- claim_update: supported at component level; end-to-end realized saving remains to be measured.
- baseline_relation: identical eager q and FP32 scale bits for all tested inputs.
- failure_mode: none in the accepted domain; launcher rejects non-eligible shapes to eager.
- next_action: run one hot same-seed launcher A/B and compare output hash plus total/denoise timing.
- artifacts: source, DSO, build logs, full benchmark JSON, robustness JSON, launcher candidate, and patch are in this directory.
