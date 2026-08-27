# Exactness-Gated Kernel Substitution: Deploying and Accelerating a 15-Second Audio-Video Diffusion Transformer on Dual Hygon DCUs

**[AUTHOR NAME]**  
**[AFFILIATION]**  
[TODO: verify] author name and affiliation.

## Abstract

15-second audio-video Diffusion Transformers spend most of a two-accelerator node on dense 3D attention and the INT8 conditioning path. USP, PipeFusion, and SwiftFusion address this on NVIDIA and AMD GPU stacks; a 2026-08-27 search found no comparable end-to-end study on Hygon DCU (gfx936).

We deploy a MiniMax H3-class 15-second audio-video DiT on a public HPC cloud node with two Hygon DCU accelerators (gfx936, 64 GB HBM each). The reusable contribution is *exactness-gated kernel substitution*: every arithmetic replacement is bitwise-exact, numerically-bounded, or quality-gated, and is bound at runtime by a shape whitelist, an audit hash, and automatic fallback.

On 608×352×15s, wall-clock falls from 1061.6s on one DCU to 116.3s on two (9.1×; 1.8× from sequence parallelism, the rest from kernels and pipeline). On 1344×768×15s (109099 tokens, 20 steps) we measure 65.3s/step and 23 min 58 s end-to-end, 6.5× faster than a 470s/step fallback. FlashAttention at 109k tokens takes 0.994s per DCU with error 8.3×10⁻⁵ versus a 1.9×10⁻⁴ baseline; six INT8 M shapes have 0 mismatches. Dense attention alone is ~52s/step and ≥17.3 min for 20 steps, so a 10-minute two-DCU target needs more accelerators or lossy methods, which we refuse.

**Keywords:** diffusion transformer, sequence parallelism, Hygon DCU, FlashAttention, INT8 kernels, quality gates, exactness

---

## 1. Introduction

A 15-second audio-video DiT is a bad fit for a two-accelerator node if the serving stack treats the model as a generic Transformer. MiniMax H3-class models attend jointly over video latents and audio tokens with 3D full attention, and they ship an INT8 checkpoint rather than a pure bf16 graph. At 1344×768×15s the sequence is 109099 tokens. Quadratic attention, a 20-step sampler, a VAE that must emit 362 frames, and a shape-sensitive INT8 conditioning path dominate wall-clock time. The question this report answers is not whether sequence parallelism can split that sequence—USP, PipeFusion, and SwiftFusion have already shown that it can, on other accelerators—but how to *replace* the slow arithmetic on a Hygon DCU without quietly changing the picture or the soundtrack.

That replacement problem is easy to get wrong. A fused attention kernel that is faster on one shape may NaN on another. An INT8 GEMM epilogue that is bit-identical on the 608×352 M values used in production can fall off its whitelist at high resolution and spend hours in a fallback path that is correct but unusable. A distributed run that is not bitwise-identical to a single-DCU canonical can still be a valid serving result, but only if the media contract and a same-seed quality comparison say so in public, rather than behind an average score. The failure mode we wanted to avoid is the usual one in accelerator bring-up: a kernel is merged because it is faster, the first out-of-sample shape is wrong, and there is no recorded gate that would have refused it.

Public sequence-parallel DiT systems do not close the gap. USP [1], PipeFusion [2], and SwiftFusion [3] build on FlashAttention [4, 5], DeepSpeed-Ulysses [6], and Ring Attention [7], all on CUDA or ROCm GPU toolchains, as do public HunyuanVideo-plus-xDiT multi-node accounts on official AMD ROCm. On Hygon DCU, a 2026-08-27 search found operator-level notes (GEAK on gfx928, hipprof on gfx936) but not a systematic audio-video DiT serving study. [TODO: verify] those blog claims; they are outside the allowed reference list and are not cited below.

This report therefore makes three claims, and only these.

1. **First-hand two-DCU measurements** of a MiniMax H3-class audio-video DiT, including native stereo audio, on a public HPC cloud node with two Hygon DCU accelerators (gfx936, 64 GB HBM each), at 608×352 and at 1344×768, for 5s, 10s, and 15s, at 20 sampling steps.

2. **Exactness-gated kernel substitution.** Replacements are bitwise-exact, numerically-bounded, or quality-gated; runtime enforcement is a shape whitelist, an audit hash, and automatic fallback. New shapes are added only after the class-appropriate test. The 2026-08-27 high-resolution campaign is the example: the FlashAttention cap `seq ≤ 50000` and the INT8 M whitelist `{11819, 12055, 12280}` were software gates, not hardware limits, and were extended only after validation.

3. **A physical lower bound** for 15-second 1344×768 dense attention on two DCUs. One FlashAttention forward at 109k tokens costs 0.994s per DCU; about 50 invocations per step cost about 52s of attention alone; 20 steps cost at least 17.3 min. A 10-minute target on two cards is not an implementation bug.

[TODO: verify] MiniMax H3 terms allow a third-party benchmark that does not redistribute weights. [TODO: verify] public HPC cloud terms allow publishing node-class specifications and performance numbers. Until both are confirmed, the hardware string stays at the node class above; host names, ports, and internal paths are omitted.

---

## 2. Background

### 2.1 Audio-video DiT inference

MiniMax H3, in the configuration we serve, uses 3D full attention over a joint video-latent and audio-token sequence, ships an INT8 checkpoint, and is sampled for 20 steps in every experiment here. We do not claim a layer count, parameter count, or hidden size; those figures are not in the measurement record. [TODO: verify] H3 architectural hyperparameters if a later revision needs them.

The serving profiles that matter are two resolutions and three durations.

- **Default production profile.** 608×352, 15s, 24 fps, 362 frames, H.264 `yuv420p`, AAC audio at 32000 Hz, two channels, container duration 15.083333 s within ordinary mux rounding. This is also the hard media contract against which quality gates run (Section 3.3 and Section 5.3).
- **High-resolution profile.** 1344×768, same 24 fps and 362 frames at 15s, H.264 plus AAC. Measured token lengths on the two-DCU Ulysses split are 37747 (5s), 73423 (10s), and 109099 (15s). A completed 15s clip was 15.084 s long.

Image-to-video (I2V) and first-last-frame (FLF) conditioning are both in the serving path. H3 keyframes are conditioning anchors, not pixel-copy constraints: a compressed frame 0 or last frame need not equal the input pixels. Quality comparisons therefore use the model's own preprocessing and, when a same-seed canonical exists, decoded RGB/PCM rather than a prompt-level aesthetic score.

### 2.2 Hygon DCU gfx936 and the DTK stack

All measurements use a public HPC cloud node with two Hygon DCU accelerators (gfx936, 64 GB HBM each). The software stack is the vendor DTK toolchain with a HIP/PyTorch serving graph. [TODO: verify] DTK, HIP, and PyTorch versions; they were not recorded in the outline or campaign notes used for this draft.

gfx936 is not a drop-in CUDA device. FlashAttention [4, 5] does not ship a supported gfx936 backend in the upstream tree we started from. INT8 GEMM epilogues that are correct for one M dimension are not automatically correct for another. Allocator behavior under a 362-frame 1344×768 VAE decode is different from the 608×352 path that had already been in production. These are the reasons a substitution gate has to live in the launcher rather than in a one-off kernel patch.

### 2.3 Sequence parallelism: Ulysses, Ring, USP

DeepSpeed-Ulysses shards the sequence, all-to-alls into a per-rank head slice that holds the full sequence, runs attention, and all-to-alls back [6]. Ring Attention keeps a sequence shard and rotates KV blocks around a ring [7]. USP (Unified Sequence Parallelism) combines the two so that a deployment can pick a Ulysses degree, a ring degree, or a product of both [1]. PipeFusion pipelines DiT inference at the patch level [2]. SwiftFusion targets scalable sequence-parallel DiT inference on GPUs [3].

With two accelerators the natural Ulysses degree is two: each DCU holds half the heads with the full sequence after the QKV all-to-all. That is the configuration used throughout. We do not claim a new collective. The 1.8× parallelism factor in Section 5.1 is less than 2× because all-to-alls, pipeline bubbles, and non-attention work do not scale. [TODO: verify] the measurement protocol behind the 1.8× attribution.

---

## 3. Exactness-Gated Kernel Substitution

The method is a policy for replacing arithmetic in a serving graph. It is deliberately conservative. A substitution that is faster on the shapes we have seen is still refused on a shape we have not tested. The cost of that conservatism is the subject of Section 5.2 and Section 6.1: the 10s and 15s high-resolution paths spent hours in fallback until the whitelist caught up with the kernel.

### 3.1 Three classes of substitution

Every candidate replacement is labeled with exactly one class before it is eligible for admission. The class determines the test, not the other way around.

**Bitwise-exact (Class I).** The substitution must match a frozen reference tensor-for-tensor, with zero mismatches, on the shapes that will be whitelisted. This is the class of the INT8 path: the row-wise quantizer, the GEMM epilogue, and related integer epilogue rewrites. On a DCU, an INT8 kernel that is correct for M ∈ `{11819, 12055, 12280}` (the 608×352 conditioning shapes) is not thereby correct for the high-resolution M values. Bitwise identity is cheap to check and expensive to assume. We do not treat a close fp16/bf16 reconstruction as Class I.

**Numerically-bounded (Class II).** The substitution is a floating-point kernel whose output is not required to be bit-identical to a naive matmul reference. FlashAttention is the prototype: tiling, online softmax, and bf16 rounding produce a different bit pattern than an unfused attention, even when both are “exact” in the FlashAttention sense [4, 5]. Admission requires (i) all finite values, no NaNs, and (ii) a sampled error no worse than a previously audited baseline of the *same* kernel family. For the gfx936 FlashAttention forward, the audited baseline is sequence length 23638, maximum sampled error 1.9×10⁻⁴, attributed to pure bf16 rounding. A long-sequence run is admitted only if it stays inside that envelope. At 109099 tokens the sampled maximum error was 8.3×10⁻⁵, which is below the baseline, and there were no NaNs. Class II is not a license to ship a new approximation. It is a license to ship a different *implementation* of the same bf16 attention, with a recorded error bar.

**Quality-gated (Class III).** The substitution changes the serving graph in a way that is not bitwise-equivalent to a single-DCU canonical, even when every kernel in the graph is Class I or Class II. Ulysses all-to-alls, rank scheduling, and any numerical reduction that is associated rather than strictly sequential belong here. Distributed inference of this model is **not** claimed to be bitwise-equivalent to single-rank inference, and it is **not** claimed to be mathematically lossless. Admission is the hard media contract plus a same-seed comparison against a canonical single-rank clip, plus temporal and audio checks (Section 3.3). One seed is evidence about that sample, not a universal neutrality proof.

The classes are exclusive at admission: a failed bitwise test is not rescued by SSIM, a failed error envelope is not rescued by a contact sheet, and a distributed pipeline that passes kernel tests still has to pass Class III.

### 3.2 Runtime gate: whitelist, audit hash, fallback

Admission is a bring-up-time act. Enforcement is a runtime gate in the launcher.

Let `op` be a named substitution (FlashAttention forward, INT8 quantizer, INT8 epilogue, …), `shape` the dynamic dimensions that affect correctness (sequence length for attention; M for the INT8 path), and `artifact` the binary or source snapshot that implements `op`.

```
SUBSTITUTE(op, shape, artifact):
    if sha256(artifact) not in AUDIT[op]:
        return REFERENCE(op, shape)
    if shape not in WHITELIST[op]:
        return REFERENCE(op, shape)
    return ARTIFACT(op, shape, artifact)
```

Three properties follow. Unknown code does not run: the audit set is SHA-256 hashes of admitted artifacts, and a rebuilt binary does not inherit a previous admission; v13/v14 high-resolution *candidates* were not allowed to overwrite an already-audited hash. Unknown shapes do not run: FlashAttention was gated at `seq ≤ 50000` and INT8 M at `{11819, 12055, 12280}`; 10s (73423 tokens) and 15s (109099) missed the attention gate, and high-resolution M ∈ `{18873, 18874, 36711, 36712, 54549, 54550}` missed the INT8 set. Fallback is mandatory: the launcher does not skip the op or try a “close” shape; it runs the reference. The hour-scale 10s/15s high-resolution runs before 2026-08-27 were this rule working, and they are why the conservatism had a cost (Section 6.1). The audit set and whitelist are data, which is what makes a substitution reviewable on an ephemeral cloud node.

### 3.3 Quality gates for Class III

Class III uses the media and semantic contract recorded in the quality-gate notes, not a single scalar.

**Hard media contract** (default 15-second profile): video H.264, `yuv420p`, 608×352, 24 fps, 362 frames; audio AAC, 32000 Hz, two channels; container duration 15.083333 s within mux rounding; every frame decodes; the audio stream is non-empty. High-resolution 15s clips additionally have to be 1344×768, 362 frames, H.264+AAC; the completed sample was 15.084 s.

**Conditioning adherence.** First/middle/last contact sheets are inspected against the same preprocessing the node applies at serving time. I2V and FLF are not baselines for each other.

**Same-seed canonical comparison.** When backend scheduling allows, decoded RGB and PCM are compared for identity. When it does not, the difference is documented, and comparison moves to endpoint adherence, temporal health, and semantics—not a single average score. Audited one-seed SSIM/PSNR values are in Section 5.3. I2V and FLF are not baselines for each other; FLF is scored on first/last-frame anchors as well as the whole clip.

**Temporal and audio checks.** Reject black or corrupt frames; flag long runs of nearly identical adjacent frames; inspect the first and last 0.5 s for hard cuts, anchor snapping, or sudden geometry changes. Audio must have finite PCM, nonzero RMS, stereo present; flag clipping, large DC offset, unexpected silence. Numeric presence of rain/footsteps/train/umbrella cues is not semantic adherence; those cues are listened for.

**Claim strength.** One seed proves that sample. At least three representative seeds are required before a mode is claimed generally quality-neutral. Kernel-level bitwise tests support Class I substitutions and do not replace media evaluation of the distributed pipeline.

### 3.4 Admission procedure

Bring-up of a new shape or a new artifact follows the same order every time.

1. Classify the substitution (I / II / III). A change that includes a distributed graph is at least Class III, even if its kernels are I or II.
2. Run the class-appropriate harness on the *target* shape, not on a proxy shape. Class I: mismatch count. Class II: NaN check and sampled error against the audited baseline. Class III: media contract, contact sheets, and same-seed metrics.
3. Only then add the shape to `WHITELIST[op]` and, if the code changed, add `sha256(artifact)` to `AUDIT[op]`.
4. Ship behind the runtime gate of Section 3.2. Do not delete the reference path.

The order is the method. Whitelist-first is how a 50000-token attention cap and a three-entry M set silently tax a high-resolution launch by hours. Validate-then-whitelist is how those caps were raised on 2026-08-27 without changing the quality claim.

### 3.5 Case study: the 2026-08-27 high-resolution whitelist

The 1344×768 campaign is the method in miniature. The hardware was the same two-DCU node; the sampler was 20 steps; the sequence parallel degree was two. Three durations were in scope: 5s (37747 tokens), 10s (73423), 15s (109099). Two independent software caps, not the gfx936 peak, blocked the 10s and 15s paths.

**Cap A: FlashAttention `seq ≤ 50000`.** The launcher used this inequality as a standard-layout admission test. At 5s, 37747 ≤ 50000, so attention substitution already fired (old path 11.06 s/step versus 10.3 s/step after the rest of the campaign). At 10s and 15s the cap refused the kernel and the run fell back. Direct validation of the gfx936 FlashAttention forward at 109099 tokens showed no NaNs and a sampled maximum error of 8.3×10⁻⁵, below the 1.9×10⁻⁴ envelope at sequence 23638. The kernel was therefore Class II-admissible at the 15s length. The whitelist was raised to 131072 as a v13 candidate. The cap that had looked like a hardware limit was a constant in the launcher.

**Cap B: INT8 M ∈ `{11819, 12055, 12280}`.** Those three values are the 608×352 conditioning shapes. High-resolution M values are 18873/18874 (5s), 36711/36712 (10s), and 54549/54550 (15s). All six were run through the Class I harness for the quantizer and the epilogue; mismatch count was 0 on every shape. They were then added as a v14 candidate. The INT8 path at high resolution is bitwise-exact on those M values, not “close.”

The two caps interact. Attention substitution without INT8 substitution still leaves the conditioning path on the reference GEMM; INT8 substitution without attention substitution still leaves 109k-token attention on the unfused path. End-to-end 15s high-resolution serving needs both. After both admissions, 15s high-resolution sampling ran at 65.3 s/step (1306 s in the sampler), with a cold end-to-end wall-clock of 23 min 58 s, against a fallback of 470 s/step (about 2.6 h). That 6.5× is a whitelist effect as much as a kernel effect: the kernels were already healthy; the gate had not been told.

After the extension, quality claims stay those of the existing production path: long-sequence FlashAttention is the same bf16 family inside the audited envelope; INT8 expansion is bitwise-exact; Class III media gates are unchanged. The campaign does not add a lossy approximation. Two operational rules: the refused M must appear in the first fallback log after a clean start (reason-level de-duplication can hide later 10s values), and candidate launchers must not overwrite the audited hash.

---

## 4. System

The substitution method sits inside a two-DCU serving stack. This section records the pieces that are in the measurement path. Where a mechanism is named in the design notes but not microbenchmarked, we describe it as deployed mechanism and do not attach a speedup. [TODO: verify] communication/compute overlap fractions, VAE stream occupancy, and copy elision counts; none are in the input measurements.

### 4.1 Dual-DCU Ulysses sequence parallelism

Ranks equal DCUs equal two. The sequence is split Ulysses-style [6]: QKV are all-to-alled so that each rank holds the full sequence for a subset of heads, attention (the Class II FlashAttention forward) runs locally, and the output is all-to-alled back. Audio and video tokens travel in the same sequence. Two-rank Ulysses fits 109099-token attention in 64 GB HBM per DCU; it does not make 20 dense steps fast (Section 5.5).

### 4.2 QKV head-chunk communication overlap

The QKV all-to-all is chunked along the head dimension so that communication of chunk *i*+1 can overlap attention on chunk *i*. This is ordinary overlap, not a new collective. We do not report an isolated speedup for it. It is part of the “pipeline” remainder in the 9.1× decomposition of Section 5.1.

### 4.3 Streaming VAE encode and decode

A 15-second clip is 362 frames. Materializing a full 1344×768×362 activation for VAE decode on a 64 GB device is how the campaign ran out of memory after a successful 20-step sample (Section 6.1). The serving path encodes and decodes in a stream rather than as one 362-frame tensor. Streaming is a Class III-adjacent graph change: it must not drop frames, must preserve order, and must still satisfy the media contract. It is not a bitwise claim against a single batched VAE call.

### 4.4 Zero-copy inverse transform

Inverse layout transforms after attention are written to avoid an extra device-to-device copy where the HIP graph allows it. No isolated microbenchmark is claimed.

### 4.5 Allocator anti-fragmentation

The 15s 1344×768 VAE decode failed with a 4.18 GB allocation on a 64 GB device after the sampler had finished—a fragmentation failure, not a peak-HBM failure. The serving environment therefore sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for that profile. High-resolution 15s is specified to require this setting. It is a configuration gate, in the same spirit as the kernel whitelist: the default allocator is the reference; the expandable-segment path is admitted for the profile that has demonstrated the failure.

The stack does not skip sampler steps, cache residuals, or replace bf16 attention with quantized attention (Section 7).

---

## 5. Evaluation

Unless noted, every run uses a public HPC cloud node with two Hygon DCU accelerators (gfx936, 64 GB HBM each), Ulysses degree 2, 20 sampler steps, and native stereo AAC. Single-DCU numbers are the same node class with one accelerator. We do not mix resolutions in a speedup, and we do not compare I2V to FLF as if one were the other's baseline.

### 5.1 Default profile: 608×352×15s

Table 1 is the headline production-resolution result.

**Table 1.** 608×352, 15 s, 20 steps.

| Configuration | Wall-clock | Speedup vs single DCU |
| --- | ---: | ---: |
| Single DCU, reference path | 1061.6 s | 1.0× |
| Dual DCU, this stack | 116.3 s | 9.1× |
| Attributed to sequence parallelism | — | 1.8× |
| Attributed to kernels and pipeline | — | remainder of 9.1× |

116.3 s is end-to-end wall-clock, not sampler-only. The 9.1× factor is 1061.6 / 116.3. The 1.8× parallelism factor is the portion attributed to the two-rank Ulysses split; the rest is kernel substitution (Class I INT8, Class II FlashAttention) plus the pipeline work in Section 4. [TODO: verify] an isolated dual-DCU-without-kernels wall-clock if a revision wants to replace “remainder” with a measured second factor.

This is the profile the media contract in Section 3.3 is written against, and the profile whose INT8 M values `{11819, 12055, 12280}` were already on the whitelist before the high-resolution campaign.

### 5.2 High-resolution profile: 1344×768

Table 2 is the 2026-08-27 campaign. Token counts are measured sequence lengths on the two-rank split. “Sampling” is 20 × step time. End-to-end includes VAE and mux. The 15s cold number is a measured wall-clock, not 20 × 65.3 s.

**Table 2.** 1344×768, dual DCU, 20 steps.

| Duration | Tokens | Step time | Sampling | End-to-end | Prior path |
| --- | ---: | ---: | ---: | --- | --- |
| 5 s | 37747 | 10.3 s/step | ~206 s | ~4.1 min (hot) | 11.06 s/step |
| 10 s | 73423 | 31.2 s/step | ~624 s | ~11.5 min (hot) | FlashAttention refused; hour-scale |
| 15 s | 109099 | 65.3 s/step | 1306 s | 23 min 58 s (cold); ~22.5 min (hot) | 470 s/step ≈ 2.6 h |

The 15s clip was inspected as a finished container: 15.084 s, 1344×768, H.264+AAC, 362 frames.

Several comparisons in Table 2 are easy to misread.

- The 5s prior path already had FlashAttention (37747 ≤ 50000). The drop from 11.06 to 10.3 s/step is the rest of the stack, including Class I INT8 on the new M pair, not a 9× attention miracle. [TODO: verify] a per-op breakdown of that 0.76 s/step.
- The 10s and 15s prior paths are fallback paths, dominated by the refused fused attention. 470 s/step at 15s is the software gate of Section 3.5, not gfx936 peak.
- 6.5× at 15s is end-to-end (23 min 58 s versus about 2.6 h), not 470 / 65.3. The cold 23 min 58 s is what an ephemeral node shows on first run; ~22.5 min is the hot figure.

### 5.3 Quality

Table 3 is the audited one-seed comparison against a canonical single-rank run. It is Class III evidence for the default 15-second media contract. It is not a 1344×768 SSIM table; the high-resolution campaign claims quality *parity of method* (same bf16 attention family, bitwise INT8, same media gates), not a new SSIM at 1344×768. [TODO: verify] a same-seed 1344×768 I2V/FLF SSIM/PSNR if a later revision wants a high-resolution quality table.

**Table 3.** Audited one-seed quality versus canonical single-rank.

| Mode | Whole-video YUV SSIM | Whole-video PSNR | First-frame SSIM | Last-frame SSIM |
| --- | ---: | ---: | ---: | ---: |
| I2V | 0.953625 | 32.841 dB | — | — |
| FLF | 0.829264 | 26.916 dB | 0.963506 | 0.956163 |

I2V is close to the canonical in the whole-video average. FLF preserves the two requested anchors (first/last SSIM 0.963506 / 0.956163) and fails to match the middle trajectory; whole-video SSIM 0.829264 and PSNR 26.916 dB are the record of that divergence, not a hidden loss in the kernels. We do not describe the distributed pipeline as lossless, and we do not average I2V and FLF into one “quality score.”

Media-contract checks (black frames, freeze / near-duplicate runs, audio stream present with finite PCM and nonzero RMS, H.264+AAC container) passed on the audited samples. Semantic cue listening (rain, footsteps, train, umbrella on the prompts that request them) is part of the gate; we do not reduce it to a number that is not in the record.

Per Section 3.3, one seed is not general quality neutrality. A three-seed claim is future work.

### 5.4 Kernel microbenchmarks

Table 4 is the Class I / Class II evidence that justified the 2026-08-27 whitelist extension.

**Table 4.** Kernel checks used for admission.

| Test | Shape | Result | Admission envelope |
| --- | --- | --- | --- |
| FlashAttention forward latency | seq = 109k tokens | 0.994 s per DCU | performance, not a gate |
| FlashAttention sampled max error | seq = 109k tokens | 8.3×10⁻⁵ | ≤ 1.9×10⁻⁴ at seq = 23638; no NaNs |
| INT8 quantizer + epilogue | M ∈ {18873, 18874, 36711, 36712, 54549, 54550} | 0 mismatches on all six | bitwise (Class I) |

The six M values are the 5s/10s/15s high-resolution pairs. Combined with `{11819, 12055, 12280}` already on the 608×352 whitelist, they are the INT8 shapes this report is willing to serve. The 109k FlashAttention error is *inside* the audited bf16 envelope, not a claim that long sequence is “more exact.”

### 5.5 Physical lower bound on two DCUs

The 15s 1344×768 path is attention-bound in a way a kernel engineer cannot paper over. Table 5 turns the 0.994 s forward into a floor.

**Table 5.** Dense-attention floor, 1344×768×15s, 20 steps, two DCUs.

| Quantity | Value |
| --- | --- |
| FlashAttention forward, seq = 109k | 0.994 s per DCU |
| Attention invocations per sampler step | ~50 |
| Pure attention per step | ~52 s |
| Pure attention, 20 steps | ≥ 17.3 min |
| Measured step time (includes non-attention) | 65.3 s |
| Measured 15s end-to-end, cold | 23 min 58 s |
| 10-minute product target on two DCUs | infeasible under dense 20-step attention |

50 × 0.994 s = 49.7 s; the campaign records ~52 s/step of attention. Twenty steps at 52 s is 1040 s ≈ 17.3 min, which is already above a 10-minute target *before* INT8, VAE, collectives, and mux. Measured 65.3 s/step is consistent with that floor plus non-attention work. The 23 min 58 s end-to-end is not a kernel bug, and shaving the 13 s/step of non-attention cannot produce a 10-minute clip.

Back-of-envelope scalings recorded in the campaign, *not measured*:

- 4 DCUs: ~13 min, still above 10 min.
- 8 DCUs: ~7–8 min, contingent on extending Ulysses (or USP [1]) and re-validating collectives under Class III.
- Fewer sampler steps, distillation, or sparse / quantized attention: potentially under 10 min, all lossy, all requiring a new Class III campaign. We refuse them for this report.

On this node class, 15s at 1344×768 is a roughly 24-minute job; advertising 10 minutes would be a false claim.

---

## 6. Lessons, Pitfalls, and Limitations

### 6.1 Pitfalls

**VAE decode fragmentation at 15s high resolution.** After a successful 20-step sample, 362 frames at 1344×768 failed to allocate 4.18 GB on a 64 GB DCU. Peak-capacity arithmetic (64 GB ≫ 4.18 GB) does not apply once the sampler has fragmented the pool. The admitted fix is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on that profile. High-resolution 15s without this setting is a known broken configuration, not an open bug.

**Conservative whitelist as a performance bug.** The FlashAttention `seq ≤ 50000` cap and the three-entry INT8 M set were correct gates with stale constants. They converted a 65.3 s/step kernel path into a 470 s/step fallback. The method in Section 3 is how we want that story to end (validate, then add). The lesson for operators is that a whitelist is part of the performance surface: treating it as a one-time safety switch guarantees that the first out-of-sample resolution will look like a hardware regression.

**Logs that hide the refused shape.** Reason-level de-duplication dropped the exact 10s M values on subsequent workers. The first fallback line after a clean start is the one that can be copied into the whitelist. A Class I gate whose reason string does not include M is not operable.

**`pgrep` self-matching on ephemeral nodes.** Process watchdogs that match on a short name will match themselves when the watchdog's command line contains that name. On a cloud node that is created for a campaign and deleted after, this shows up as a worker that is “restarting” with no kernel fault. The serving scripts have to match on a specific executable path or a pidfile, not on a substring shared with the watchdog. [TODO: verify] incident identifiers if a camera-ready version wants a dated example; the outline records the pitfall, not a ticket.

**Reproducibility on a machine that will be reclaimed.** Validation scripts, whitelist constants, audit hashes, and the Table 2 environment have to leave the node together. Candidate launchers were kept beside the audited launcher rather than edited in place.

### 6.2 Limitations

**Single model.** Every number is MiniMax H3-class audio-video DiT. We do not claim transfer to other DiTs, to text-to-image, or to an H3 configuration that does not use 3D full attention and an INT8 checkpoint.

**Single hardware class.** Every number is a public HPC cloud node with two Hygon DCU accelerators (gfx936, 64 GB HBM each). We do not claim CUDA, ROCm GPU, gfx928, or a different HBM capacity. The 17.3 min floor is a two-DCU dense-attention floor at 109k tokens for this FlashAttention port; it is not a vendor peak-FLOP claim.

**Distributed serving is not bitwise-equivalent to single-rank.** Class III exists because Ulysses and rank scheduling do not preserve bit identity. I2V whole-video SSIM 0.953625 / PSNR 32.841 dB and FLF whole-video SSIM 0.829264 / PSNR 26.916 dB are the measured distance to a same-seed canonical, not rounding error in a proof of equivalence. Anchor-preserving FLF (first/last SSIM 0.963506 / 0.956163) is compatible with a divergent middle. Anyone quoting this report as “lossless SP” is misquoting it.

**Sample size.** Quality numbers are one seed per mode. Section 3.3 already forbids a general quality-neutral claim on that basis. Step times in Table 2 are campaign measurements on this node class, not a multi-node statistical study. [TODO: verify] additional seeds and an independent rerun on a second node before a camera-ready “typical” claim.

**Unmeasured pieces and estimates.** Head-chunk overlap, streaming VAE, and zero-copy inverse transforms are not isolated in Tables 1–2, so 9.1× and 6.5× must not be assigned entirely to FlashAttention. The 4-DCU ~13 min and 8-DCU ~7–8 min figures in Section 5.5 are arithmetic, not experiments.

**Licensing and venue.** Model and cloud-platform terms remain as in Section 1. The FlashAttention gfx936 forward is a fork of Tri Dao's BSD-3 code; the upstream copyright notice is retained. [TODO: verify] the fork tree still carries that notice after sanitization.

---

## 7. Related Work

**Sequence-parallel DiT serving.** USP unifies Ulysses and ring parallelism for long-context generation and is the conceptual parent of “choose a Ulysses degree on two devices” [1]. PipeFusion pipelines DiT inference at patch granularity and is the reference for patch-level pipeline parallelism [2]. SwiftFusion studies scalable sequence-parallel DiT inference on GPUs [3]. DeepSpeed-Ulysses and Ring Attention are the two communication primitives [6, 7]. We use two-rank Ulysses and do not claim a better collective. The difference is the accelerator (gfx936 DCU), the workload (joint audio-video, INT8 checkpoint), and the admission gate around the kernels that make Ulysses usable at 109k tokens.

**IO-aware attention.** FlashAttention and FlashAttention-2 are the algorithm we port [4, 5]. Class II is how that port is admitted on gfx936: not bit-identical to unfused attention, bounded by an audited bf16 envelope, refused when the sequence length is not on the whitelist.

**Lossy step caches.** Serving literature on DiTs includes caches that skip or reuse computation across sampler steps (commonly discussed under names such as TeaCache and MagCache). They reduce step count or step cost by changing the numerical trajectory. That is a Class III change with a different, weaker contract than ours, and we do not use it. [TODO: verify] formal citations; they are omitted because they are outside the allowed reference list.

**Quantized attention.** Kernels that quantize QK/V (commonly discussed under names such as SageAttention) are a different Class II-or-worse substitution: they change attention arithmetic, not just its tiling. Our INT8 work is on the *conditioning* GEMM path (quantizer and epilogue, bitwise-exact on whitelisted M). Attention remains bf16 FlashAttention. We do not claim a quantized attention result, and we do not use one to break the 17.3 min floor. [TODO: verify] formal citations; omitted for the same reason.

**DCU and ROCm notes.** Operator-level DCU posts and AMD ROCm HunyuanVideo-plus-xDiT writeups (2026-08-27 search) support the gap claim in Section 1; they are not baselines and are not in References.

---

## 8. Conclusion

A two-DCU Hygon gfx936 node can serve a 15-second MiniMax H3-class audio-video DiT, with native stereo audio, at 608×352 in 116.3 s and at 1344×768 in a measured 23 min 58 s, if the serving stack is allowed to replace arithmetic under an explicit exactness gate. The 9.1× at 608×352 and the 6.5× at 1344×768×15s are the product of that gate plus two-rank Ulysses, not of a new sequence-parallel algorithm. The kernels that matter are a gfx936 FlashAttention forward whose 109k-token error (8.3×10⁻⁵) sits inside an audited bf16 envelope (1.9×10⁻⁴ at sequence 23638), and an INT8 quantizer/epilogue whose six high-resolution M shapes mismatch the reference zero times. The same measurements close a product argument: 20 dense attention steps at 109k tokens cannot meet a 10-minute budget on two DCUs, because attention alone is already 17.3 min.

Exactness-gated substitution is the part we expect to reuse on the next model or the next DCU generation. Classify the replacement, test the class, write the shape down, hash the artifact, keep the fallback. The 2026-08-27 high-resolution campaign is the example of the rule being applied after it had been paid for: the fused kernels were already healthy at 109k tokens and at the high-resolution M values; a `seq ≤ 50000` constant and a three-entry M set were the reason a 24-minute job had been a 2.6-hour job.

Limits are intentional: one model, one node class, one seed per quality mode, a distributed path that is not bitwise-equivalent to single-rank, and 4-/8-DCU figures that are arithmetic. A camera-ready revision should add authors, licenses, stack versions, a three-seed quality table, and, if a 10-minute target remains, a measured multi-DCU or explicitly lossy path that re-runs Class III.

---

## References

Only the identifiers listed in the outline are used. Titles and authors below match `oc-cite verify` resolution on 2026-08-27 (resolved, not flagged as retracted). NeurIPS 2025 for PipeFusion is taken from the outline.

[1] Jiarui Fang and Shangchun Zhao. USP: A Unified Sequence Parallelism Approach for Long Context Generative AI. arXiv:2405.07719, 2024.

[2] Jiarui Fang, Jinzhe Pan, Aoyu Li, Xibo Sun, and Jiannan Wang. PipeFusion: Patch-level Pipeline Parallelism for Diffusion Transformers Inference. arXiv:2405.14430, 2024. (NeurIPS 2025.)

[3] Jiacheng Yang, Jun Wu, Yaoyao Ding, Zhiying Xu, Yida Wang, and Gennady Pekhimenko. SwiftFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs. arXiv:2601.20273, 2026.

[4] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135, 2022.

[5] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691, 2023.

[6] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. arXiv:2309.14509, 2023.

[7] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889, 2023.
