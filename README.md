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

## License

- 本仓库自研代码(kernels 中除 flash-attention-gfx936 外的全部、launcher/、validation/、benchmarks/)以 **Apache-2.0** 发布(根目录 LICENSE)。
- `kernels/flash-attention-gfx936/` 派生自 [Tri Dao FlashAttention](https://github.com/Dao-AILab/flash-attention),遵循其 **BSD-3-Clause**,原 LICENSE/AUTHORS 保留在该目录内。
- **不包含** ComfyUI(GPL-3.0,请自行安装;launcher 在运行时对其打补丁)与 MiniMax H3 权重(受 MiniMax H3 Community License 约束,含地域限制——美国/欧盟/英国/韩国不在授权区,请自行核对后从[官方渠道](https://huggingface.co/MiniMaxAI/MiniMax-H3)获取)。

## 环境

实测环境:2×Hygon DCU gfx936(64 GB HBM)、DTK 26.04(dcc 25.10.0 / clang 17)、PyTorch 2.9.0、HIP 6.3。

## 状态

- [x] 源码脱敏整理进本仓库
- [x] 论文全文初稿(paper/draft.md,署名 dengxuan)
- [x] LICENSE(Apache-2.0 + flash 目录 BSD-3)
- [x] GitHub 仓库创建与推送
- [x] 论文 TODO 清单消项(引用经 arXiv 官方页核对)
- [ ] LaTeX 化、arXiv 投稿
