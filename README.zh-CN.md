# dcu-video-dit-turbo

[English](README.md) | **简体中文**

**面向海光 DCU(gfx936)的长时音视频扩散 Transformer 推理加速:逐位精确门控的内核替换。**

本仓库包含技术报告 *Exactness-Gated Kernel Substitution: Deploying and Accelerating a 15-Second Audio-Video Diffusion Transformer on Dual Hygon DCUs* 背后的全部内核、门控启动层、验证脚本与基准测试([paper/draft.md](paper/draft.md),LaTeX 源码见 [paper/latex/](paper/latex/))。

在两张 gfx936 DCU(各 64 GB HBM)上,MiniMax H3 级模型生成 15 秒 608×352 成片(含原生立体声)端到端约 **116 秒**,15 秒 1344×768 成片约 **24 分钟**——相对同硬件各自的基线路径分别快 9.1× 和 6.5×;且每一处内核替换要么逐位等价、要么误差在已审计包络内、要么显式通过媒体质量门。

## 示例视频

以下短片全部由本套栈在 2×gfx936 上端到端生成(20 步采样、原生立体声、无后处理、无挑帧超分)。点击文件即可在 GitHub 上播放。

| 场景 | 规格 | 全程耗时 | 文件 |
| --- | --- | --- | --- |
| 宇宙战舰(史诗科幻) | 15 s · 1344×768 · 24 fps | ~24 min | [examples/space-battleship-15s-1344x768.mp4](examples/space-battleship-15s-1344x768.mp4) |
| 日出纸船(微距) | 15 s · 1344×768 · 24 fps | ~24 min | [examples/paper-boat-15s-1344x768.mp4](examples/paper-boat-15s-1344x768.mp4) |
| 晨雾瀑布山谷(航拍) | 5 s · 1344×768 · 24 fps | 5 min 22 s | [examples/nature-waterfall-5s-1344x768.mp4](examples/nature-waterfall-5s-1344x768.mp4) |
| 雨夜霓虹街道(都市) | 5 s · 1344×768 · 24 fps | 5 min 10 s | [examples/city-rain-night-5s-1344x768.mp4](examples/city-rain-night-5s-1344x768.mp4) |
| 风雪雪豹(野生动物) | 5 s · 1344×768 · 24 fps | 5 min 15 s | [examples/snow-leopard-5s-1344x768.mp4](examples/snow-leopard-5s-1344x768.mp4) |
| 黄金时刻海岸(无人机) | 15 s · 608×352 · 24 fps | 3 min 7 s | [examples/ocean-coast-15s-608x352.mp4](examples/ocean-coast-15s-608x352.mp4) |

上表耗时为完整作业实测时间(提示词编码、20 步采样、VAE 解码、mp4 封装与文件传输),因此略高于下方基准表中的纯采样数字。

## 实测结果

2×gfx936(64 GB),20 步,视频+音频联合生成:

| 分辨率 | 时长 | 每步 | 端到端 | 同硬件基线 |
| --- | --- | --- | --- | --- |
| 608×352 | 15 s | 5.0 s | ~116 s | 单卡 1061.6 s(**9.1×**) |
| 1344×768 | 5 s | 10.3 s | ~4.1 min | 旧路径 11.06 s/step |
| 1344×768 | 10 s | 31.2 s | ~11.5 min | 回退路径,小时级 |
| 1344×768 | 15 s | 65.3 s | 实测 23 min 58 s | 回退路径 470 s/step ≈ 2.6 h(**6.5×**) |

报告还推导了一个物理下限:在 109k token 序列长度下,两卡上仅稠密注意力就需要 ≥17.3 分钟(20 步),因此"15 秒 1344×768 压进 10 分钟"需要更多加速卡或有损手段——这不是实现差距。

## 方法一段话

推理路径上的每一处算术替换都必须通过三类门之一:**第一类(逐位精确)**——INT8 量化器/epilogue/GEMM 替换在白名单形状上必须产生完全相同的比特,由 [validation/](validation/) 中的脚本逐位验证;**第二类(数值有界)**——bf16 FlashAttention 前向移植按序列长度准入,采样误差必须落在已审计包络内(23638 token 基线 1.9e-4;109099 token 实测 8.3e-5);**第三类(质量门控)**——分布式运行与单卡不逐位等价,因此必须显式通过媒体门(同 seed SSIM/PSNR、黑帧/冻结帧检查、音轨检查),而不是被默认视为等价。形状白名单 + 审计哈希 + 自动回退,使每处替换"可拒绝"而非"尽力而为"。

## 仓库结构

- `kernels/` — HIP 内核:精确逐行 INT8 量化器、INT8 GEMM epilogue(标量与 vec4),以及 `flash-attention-gfx936/`——[FlashAttention](https://github.com/Dao-AILab/flash-attention) 的 gfx936 前向移植(BSD-3,保留上游版权声明)。
- `launcher/` — 门控启动层(形状白名单、审计哈希校验、自动回退、双卡 Ulysses 序列并行协调)。
- `validation/` — INT8 形状逐位验证与 FlashAttention 长序列数值验证。
- `benchmarks/` — 上述表格所用的基准启动器与微基准。
- `docs/` — 优化过程记录与质量门定义。
- `paper/` — 技术报告(Markdown 草稿与 arXiv LaTeX 源码)。
- `examples/` — 上方示例视频。

## 环境要求

这是研究产物,不是开箱即用的产品。复现结果需要:

- 2× 海光 DCU gfx936(64 GB),DTK 26.04(dcc 25.10.0 / clang 17),PyTorch 2.9.0 + HIP 6.3;
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)(GPL-3.0,不含在本仓库中——启动层在运行时对其打补丁);
- MiniMax H3 模型权重,从[官方发布页](https://huggingface.co/MiniMaxAI/MiniMax-H3)获取——**本仓库不含权重**。MiniMax H3 社区许可有地域限制(美国、欧盟、英国、韩国不在本地部署授权区),下载前请自行确认资格。

## 许可

- 本仓库的自研代码(除 FlashAttention fork 外的内核、launcher、validation、benchmarks)以 **Apache-2.0** 发布(根目录 [LICENSE](LICENSE))。
- `kernels/flash-attention-gfx936/` 遵循上游 **BSD-3-Clause**;原 LICENSE、AUTHORS 与逐文件版权头保留在该目录内。
- 本仓库不再分发任何模型权重与 ComfyUI 代码。

本项目为独立研究,与 MiniMax、海光或任何算力平台无隶属或背书关系。

## 引用

```bibtex
@techreport{dengxuan2026exactness,
  title  = {Exactness-Gated Kernel Substitution: Deploying and Accelerating a
            15-Second Audio-Video Diffusion Transformer on Dual Hygon DCUs},
  author = {dengxuan},
  year   = {2026},
  note   = {Technical report, https://github.com/dream-star-end/dcu-video-dit-turbo}
}
```
