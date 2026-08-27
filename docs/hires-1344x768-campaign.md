# 1344×768 高清战役(2026-08-27)

## 结论(实测,2×gfx936 BW,20 步,双卡 SP)

| 时长 | 每步 | 采样 | 端到端(热) | 旧路径 |
| --- | --- | --- | --- | --- |
| 5s (37747 tok) | 10.3 s/it | ~206 s | ~4.1 min | 11.06 s/it |
| 10s (73423 tok) | 31.2 s/it | ~624 s | ~11.5 min | flash 拒收,小时级 |
| 15s (109099 tok) | 65.3 s/it | 1306 s | 23m58s 实测(冷),~22.5 min 热 | 470 s/it ≈ 2.6h |

15s 完整成片已验证:`t2v_15s_20step_00001_.mp4`,15.084 s,1344×768,H.264+AAC,362 帧。

## 根因与修法

两个都是保守软件白名单,不是内核物理极限:

1. **FlashAttention 门禁 `seq ≤ 50000`**(launcher 标准布局判断)。内核在 109099 序列上数值健康:无 NaN,采样最大误差 8.3e-5,低于已审计 23638 基线的 1.9e-4(纯 bf16 舍入)。验证脚本:节点 `/root/h3-kernel-campaign/validate_flash_longseq.py`。放宽到 131072 → v13 候选。
2. **INT8 M 白名单 `{11819,12055,12280}`** 只含 608×352 形状。高清 M:5s=18873/18874,10s=36711/36712,15s=54549/54550。全部经 `validate_conditioning_int8_shapes_*.py` 逐位验证(quantizer/epilogue 0 mismatch)后加白 → v14 候选。

候选文件(未覆盖已审计 v12,其 sha 不变):
`/root/h3-kernel-campaign/conditioning_sp_candidate/benchmark_launcher_v14_hires_int8_candidate.py`
启动时 `H3_SP_LAUNCHER` 指向 v14,其余 env 与 v12 相同。

## 必踩坑

- **15s 高清 VAE 解码 OOM**:20 步采样完成后 362 帧 1344×768 解码碎片化 OOM(4.18GB 分配失败/64GB)。修法:启动 env 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,复跑通过。高清 15s 必须带此项。
- 10s 的精确 M 值日志会被 reason 级去重吞掉;新 worker 重启后先跑目标时长,首条 `quantizer fallback reason=M` 即精确值。

## 15s 压 10 分钟的物理账(2 卡,不降质)

单次 flash 前向 @109k = 0.994 s/卡,每步 50 块 ≈ 52 s 纯注意力;20 步纯注意力 ≈ 17.3 min > 10 min。**2 卡稠密注意力 + 20 步在物理上进不了 10 分钟**,与实现无关。可选路线:8 卡 SP(估 ~7-8 min,需扩 SP 与通信验证);减步数/蒸馏或稀疏注意力(有损,需质量门重新评估)。4 卡估 ~13 min,仍超。

## 质量声明口径

flash 长序列 = 同一 bf16 内核,误差 ≤ 已审计基线;INT8 扩白 = 逐位精确。质量口径与既有生产路径完全一致(同 seed 对单卡非逐位,媒体门照过),不引入新损失。
