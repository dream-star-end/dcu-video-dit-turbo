# 论文大纲(工作题)

**Exactness-Gated Kernel Substitution: Deploying and Accelerating a 15-Second
Audio-Video Diffusion Transformer on Dual Hygon DCUs**

形态:arXiv cs.DC 技术报告(英文),可再投国内系统会议/期刊(如 CCF 系统类、HPC China 工业 track)。

## 定位(2026-08-27 检索核实)

已有工作与空白:
- xDiT/USP(Ulysses+Ring, arXiv:2405.07719)、PipeFusion(NeurIPS 2025)、SwiftFusion(arXiv:2601.20273):DiT 推理序列并行,全部在 NVIDIA/AMD 官方生态上。
- AMD ROCm 官方博客有 HunyuanVideo+xDiT 多机实践;海光 DCU 侧只有零散算子实践博文(GEAK/gfx928、hipprof/gfx936),**没有音视频联合 DiT 端到端推理优化的系统性论文**。
- 我们的差异点不是"又一个序列并行",而是:
  1. 国产 gfx936 DCU 上音视频 DiT(视频+音频联合生成)完整推理链路的第一手系统数据;
  2. **逐位精确门控的内核替换方法论**(exactness-gated substitution):每个算术替换要么逐位等价、要么显式过质量门,配形状白名单+审计哈希+自动回退,这是可复用的工程贡献;
  3. 长序列(109k token)下的实测物理下限分析(稠密注意力算力墙)。

## 章节

1. **Introduction** — 音视频 DiT 推理成本;国产加速卡生态空白;贡献列表。
2. **Background** — MiniMax H3 架构要点(3D 全注意力、音视频联合、int8 checkpoint);gfx936/DTK 平台;Ulysses SP。
3. **Exactness-Gated Kernel Substitution(方法核心)**
   - 分类:逐位等价类(INT8 quantizer/epilogue/GEMM 换法)vs 数值等价类(bf16 flash attention,误差≤基线)vs 质量门类(分布式 vs 单卡非逐位,SSIM/PSNR+媒体契约);
   - 形状白名单 + 审计 SHA 清单 + 回退路径;白名单扩展流程(先验证后加白,2026-08-27 高清战役即案例)。
4. **System** — 双卡 Ulysses SP;QKV head-chunk 通信重叠;流式 VAE 编码;零拷贝逆变换;显存抗碎片。
5. **Evaluation**
   - 608×352×15s:单卡 1061.6s → 双卡 116.3s(9.1×,其中并行 1.8×,其余为内核+流水线);
   - 1344×768:5s=10.3s/step(~4.1min)、10s=31.2s/step(~11.5min)、15s=65.3s/step(端到端实测 23m58s,对比回退路径 470s/step≈2.6h,6.5×);
   - 质量:I2V 全片 SSIM 0.9536/PSNR 32.84dB vs 单卡 canonical;FLF 首尾帧 SSIM 0.9635/0.9562;媒体契约(黑帧/冻结/音轨)全过;
   - 内核微基准:flash@109k=0.994s/卡,误差 8.3e-5 ≤ 23638 基线 1.9e-4;INT8 六个高清 M 形状逐位 0 mismatch;
   - 物理下限:2 卡稠密注意力 20 步 ≥17.3min,10 分钟目标需 8 卡或有损手段(拒绝)。
6. **Lessons & Pitfalls** — VAE 解码碎片化 OOM(expandable_segments);pgrep 自匹配;白名单保守上限的代价;临时算力节点的可复现工程。
7. **Related Work** — xDiT/USP/PipeFusion/SwiftFusion、TeaCache/MagCache(有损缓存,划界)、SageAttention/量化注意力(划界:我们不做量化注意力)。
8. **Conclusion**。

## 引用底稿(待 oc-cite 接地校验,服务恢复后)

- USP: arXiv:2405.07719
- PipeFusion: arXiv:2405.14430 (NeurIPS 2025)
- SwiftFusion: arXiv:2601.20273
- FlashAttention: arXiv:2205.14135 / FlashAttention-2: arXiv:2307.08691
- DeepSpeed-Ulysses: arXiv:2309.14509
- Ring Attention: arXiv:2310.01889

## 硬前提(发布前必须确认)

- [ ] MiniMax H3 模型许可允许发表第三方 benchmark(不重分发权重,通常没问题,需查条款)
- [ ] SCNet 平台条款允许公开节点规格与性能数据
- [ ] flash fork 保留 BSD-3 版权声明(已确认上游 Tri Dao BSD-3)
- [ ] 署名与单位(需用户提供)
- [ ] 敏感信息清洗:节点主机名/端口/token/内部路径全部脱敏
