# 1344×768 high-resolution campaign notes (2026-08-27)

Working notes behind Section 5.2 of the technical report. All numbers were measured on 2×gfx936 (64 GB), 20 steps, dual-card Ulysses sequence parallelism.

## Measured results

| Duration | Per step | Sampling | End-to-end (hot) | Prior path |
| --- | --- | --- | --- | --- |
| 5 s (37747 tok) | 10.3 s/it | ~206 s | ~4.1 min | 11.06 s/it |
| 10 s (73423 tok) | 31.2 s/it | ~624 s | ~11.5 min | FlashAttention refused; hours |
| 15 s (109099 tok) | 65.3 s/it | 1306 s | 23 min 58 s measured (cold), ~22.5 min hot | 470 s/it ≈ 2.6 h |

A full 15-second render was verified end-to-end: 15.084 s, 1344×768, H.264 + AAC, 362 frames.

## Root causes and fixes

Both bottlenecks were conservative software whitelists, not kernel physical limits:

1. **FlashAttention gate `seq ≤ 50000`** (standard-layout admission test in the launcher). The kernel is numerically healthy at 109099 tokens: no NaNs, sampled max error 8.3e-5, below the audited 23638-token baseline of 1.9e-4 (pure bf16 rounding). Validation script: [`validation/validate_flash_longseq.py`](../validation/validate_flash_longseq.py). The cap was raised to 131072 in the v13 candidate launcher.
2. **INT8 M whitelist `{11819, 12055, 12280}`** only covered 608×352 shapes. High-resolution M values: 5 s = 18873/18874, 10 s = 36711/36712, 15 s = 54549/54550. All were bitwise-validated (quantizer/epilogue 0 mismatch) by the [`validation/validate_conditioning_int8_shapes_*.py`](../validation/) scripts before whitelisting in the v14 candidate: [`launcher/benchmark_launcher_v14_hires_int8_candidate.py`](../launcher/benchmark_launcher_v14_hires_int8_candidate.py).

Candidate launchers are separate files; the previously audited launcher is never edited in place, so its audit hash stays valid and fallback remains available. To use v14, point the launcher env var (`H3_SP_LAUNCHER`) at it and keep the rest of the environment unchanged.

## Pitfalls

- **15 s high-resolution VAE-decode OOM**: after 20-step sampling completes, decoding 362 frames at 1344×768 fails with a fragmented 4.18 GB allocation (out of 64 GB). Fix: add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the worker environment. Mandatory for 15 s at this resolution.
- The exact refused M values for 10 s get swallowed by reason-level log de-duplication. On a freshly started worker, run the target duration first; the first `quantizer fallback reason=M` line carries the exact value to whitelist.

## The physics of a 10-minute 15-second target (2 cards, no quality loss)

One FlashAttention forward at 109k tokens costs 0.994 s per card; ~50 invocations per step ≈ 52 s of pure attention; 20 steps ≈ 17.3 min > 10 min. **Dense attention at 20 steps cannot fit into 10 minutes on two cards**, independent of implementation. Options: 8-card sequence parallelism (estimated ~7–8 min, requires extending SP and validating communication); fewer steps / distillation / sparse attention (lossy, requires re-running the quality gates). A 4-card estimate is ~13 min, still over.

## Quality claim discipline

Long-sequence FlashAttention = the same bf16 kernel with error ≤ the audited baseline; the INT8 whitelist extension = bitwise-exact. The quality story is identical to the existing production path (same-seed vs single-card is not bitwise; media gates still apply); no new loss is introduced.
