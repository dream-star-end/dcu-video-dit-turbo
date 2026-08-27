# dcu-video-dit-turbo

**English** | [简体中文](README.zh-CN.md)

**Exactness-gated kernel substitution for long audio-video Diffusion Transformers on Hygon DCU (gfx936).**

This repository contains the kernels, launcher gating layer, validation harnesses, and benchmarks behind the technical report *Exactness-Gated Kernel Substitution: Deploying and Accelerating a 15-Second Audio-Video Diffusion Transformer on Dual Hygon DCUs* ([paper/draft.md](paper/draft.md), LaTeX source in [paper/latex/](paper/latex/)).

On two gfx936 DCUs (64 GB HBM each), a MiniMax H3-class model generates a 15-second 608×352 clip with native stereo audio in **~116 seconds** end-to-end, and a 15-second 1344×768 clip in **~24 minutes** — 9.1× and 6.5× faster than the respective baseline paths on the same hardware, with every kernel substitution either bitwise-exact, numerically bounded by an audited envelope, or passed through explicit media-quality gates.

## Example videos

All clips below were generated end-to-end on 2×gfx936 with this stack (20 sampling steps, native stereo audio, no post-processing, no cherry-picked upscaling). Click a file to play it on GitHub.

| Scene | Spec | Wall-clock | File |
| --- | --- | --- | --- |
| Space battleship (epic sci-fi) | 15 s · 1344×768 · 24 fps | ~24 min | [examples/space-battleship-15s-1344x768.mp4](examples/space-battleship-15s-1344x768.mp4) |
| Paper boat at sunrise (macro) | 15 s · 1344×768 · 24 fps | ~24 min | [examples/paper-boat-15s-1344x768.mp4](examples/paper-boat-15s-1344x768.mp4) |
| Waterfall valley at dawn (aerial) | 5 s · 1344×768 · 24 fps | 5 min 22 s | [examples/nature-waterfall-5s-1344x768.mp4](examples/nature-waterfall-5s-1344x768.mp4) |
| Rainy neon street at night (urban) | 5 s · 1344×768 · 24 fps | 5 min 10 s | [examples/city-rain-night-5s-1344x768.mp4](examples/city-rain-night-5s-1344x768.mp4) |
| Snow leopard in snowfall (wildlife) | 5 s · 1344×768 · 24 fps | 5 min 15 s | [examples/snow-leopard-5s-1344x768.mp4](examples/snow-leopard-5s-1344x768.mp4) |
| Ocean coastline at golden hour (drone) | 15 s · 608×352 · 24 fps | 3 min 7 s | [examples/ocean-coast-15s-608x352.mp4](examples/ocean-coast-15s-608x352.mp4) |

Wall-clock above is the full observed job time (prompt encoding, 20-step sampling, VAE decode, mp4 mux, and file transfer), which is why it is slightly above the sampler-only figures in the table below.

## Measured results

2×gfx936 (64 GB), 20 steps, joint video+audio generation:

| Resolution | Duration | Per step | End-to-end | Baseline on same hardware |
| --- | --- | --- | --- | --- |
| 608×352 | 15 s | 5.0 s | ~116 s | 1061.6 s single-DCU (**9.1×**) |
| 1344×768 | 5 s | 10.3 s | ~4.1 min | 11.06 s/step prior path |
| 1344×768 | 10 s | 31.2 s | ~11.5 min | fallback path, hours |
| 1344×768 | 15 s | 65.3 s | 23 min 58 s measured | 470 s/step fallback ≈ 2.6 h (**6.5×**) |

The report also derives a physical lower bound: at 109k tokens, dense attention alone costs ≥17.3 min for 20 steps on two DCUs, so a 10-minute 15-second 1344×768 target requires more accelerators or lossy methods — it is not an implementation gap.

## Method in one paragraph

Every arithmetic substitution in the serving path is admitted through one of three gates: **Class I (bitwise-exact)** — INT8 quantizer/epilogue/GEMM replacements must produce identical bits on whitelisted shapes, verified by the harnesses in [validation/](validation/); **Class II (numerically bounded)** — the bf16 FlashAttention forward port is admitted per sequence length only if sampled error stays within an audited envelope (1.9e-4 at the 23638-token baseline; 8.3e-5 measured at 109099 tokens); **Class III (quality-gated)** — distributed runs are not bitwise-identical to single-DCU runs, so they pass explicit media gates (same-seed SSIM/PSNR, black/frozen-frame checks, audio checks) instead of being silently assumed equivalent. Shape whitelists plus audit hashes plus automatic fallback make the substitutions refusable rather than best-effort.

## Repository layout

- `kernels/` — HIP kernels: exact rowwise INT8 quantizer, INT8 GEMM epilogues (scalar and vec4), and `flash-attention-gfx936/`, a gfx936 forward port of [FlashAttention](https://github.com/Dao-AILab/flash-attention) (BSD-3, upstream notices retained).
- `launcher/` — the gating launcher (shape whitelists, audit-hash checks, automatic fallback, dual-DCU Ulysses sequence-parallel coordination).
- `validation/` — bitwise INT8 shape validation and long-sequence FlashAttention numerical validation.
- `benchmarks/` — benchmark launchers and micro-benchmarks used for the tables above.
- `docs/` — campaign notes and quality-gate definitions.
- `paper/` — the technical report (Markdown draft and arXiv LaTeX source).
- `examples/` — the sample videos above.

## Requirements

This is a research artifact, not a turnkey product. Reproducing the results requires:

- 2× Hygon DCU gfx936 (64 GB), DTK 26.04 (dcc 25.10.0 / clang 17), PyTorch 2.9.0 on HIP 6.3;
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (GPL-3.0, not included — the launcher patches it at runtime);
- MiniMax H3 model weights, obtained from the [official release](https://huggingface.co/MiniMaxAI/MiniMax-H3) — **not included**. The MiniMax H3 Community License has territorial restrictions (the US, EU, UK, and Republic of Korea are excluded from local-deployment rights); check your eligibility before downloading.

## License

- Original code in this repository (kernels except the FlashAttention fork, launcher, validation, benchmarks) is released under **Apache-2.0** (root [LICENSE](LICENSE)).
- `kernels/flash-attention-gfx936/` follows upstream **BSD-3-Clause**; the original LICENSE, AUTHORS, and per-file copyright headers are retained in that directory.
- No model weights and no ComfyUI code are redistributed here.

This project is an independent research effort and is not affiliated with or endorsed by MiniMax, Hygon, or any compute platform.

## Citation

```bibtex
@techreport{dengxuan2026exactness,
  title  = {Exactness-Gated Kernel Substitution: Deploying and Accelerating a
            15-Second Audio-Video Diffusion Transformer on Dual Hygon DCUs},
  author = {dengxuan},
  year   = {2026},
  note   = {Technical report, https://github.com/dream-star-end/dcu-video-dit-turbo}
}
```
