# dcu-video-dit-turbo(暂定名)

Exactness-gated kernel substitution toolkit for running long audio-video
Diffusion Transformers (MiniMax H3-class) on Hygon DCU (gfx936) — dual-card
Ulysses sequence parallelism, bitwise-validated INT8 kernels, and a
long-sequence FlashAttention forward port.

实测(2×gfx936 BW 64GB,20 步,含原生立体声音频):

| 分辨率 | 时长 | 每步 | 端到端 |
| --- | --- | --- | --- |
| 608×352 | 15s | 5.0s | ~116s |
| 1344×768 | 5s | 10.3s | ~4.1min |
| 1344×768 | 10s | 31.2s | ~11.5min |
| 1344×768 | 15s | 65.3s | ~24min(实测 23m58s) |

对比:同硬件单卡基线 15s@608×352 为 1061.6s(9.1×);1344×768@15s 回退路径为
470s/step ≈ 2.6h(6.5×)。

## 计划开源内容(从生产资产脱敏整理)

- `kernels/` — HIP 内核:exact rowwise INT8 quantizer、INT8 GEMM epilogue
  (标量+vec4)、mod scale-shift、swiglu-quant;flash-attention gfx936 forward
  fork(上游 Tri Dao FlashAttention,BSD-3,保留原版权声明)。
- `launcher/` — ComfyUI 补丁层:形状白名单门控、审计 SHA 校验、自动回退、
  Ulysses SP 协调(脱敏版)。
- `validation/` — 逐位验证与长序列数值验证 harness(先验证后加白流程)。
- `benchmarks/` — 复现脚本与三档分辨率基准。
- `docs/` — 运行手册、必踩坑(VAE OOM/expandable_segments、白名单扩展流程)。
- `paper/` — 技术报告(arXiv 预印本)。

不包含:MiniMax H3 权重(请从官方渠道获取)、任何平台凭据、内部部署粘合层。

## 状态

- [x] 源码从算力节点备份(h3-kernel-campaign-src-20260827.tar.gz)
- [x] 论文大纲与文献定位(paper/OUTLINE.md)
- [ ] 源码脱敏整理进本仓库
- [ ] 论文初稿
- [ ] LICENSE 定稿(建议 Apache-2.0,flash fork 目录保留 BSD-3)
- [ ] GitHub 仓库创建与首推
