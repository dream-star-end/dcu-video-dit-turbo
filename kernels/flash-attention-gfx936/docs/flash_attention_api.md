# flash-attention算子加速库

## flash-attention简介

Flash-attention库是一款基于DCU针对attention计算的高性能算子库，提供了定长、变长，fp16/bf16/fp8，稀疏化等多种接口的实现。

## 编译安装

1. 环境依赖:

    Pytorch版本要求: pytorch2.1/pytoch2.3/pytoch2.5/pytoch2.7/pytoch2.9

    DTK版本要求: dtk23.04/dtk23.10/dtk24.04/dtk25.04/dtk26.04

2. 安装脚本:

    `python3 setup.py install`(dtk version>=23.04)

3. 编译安装包:

    `python3 setup.py install bdist_wheel`

    编译好之后，安装包在dist目录

## 安装验证

如果您想验证flash-attention是否安装正常，可通过一下命令进行验证：

```Python
python -c "import flash_attn; print(flash_attn.__version__)" #若成功打印版本信息则安装成功。
```

## Attention算子介绍

**Flash Attention**是一种加速注意力计算的方法，旨在提高计算速度和内存利用效率。它已被应用于多个知名的大型语言模型（如GPT\-3、GPT\-4、Llama2等），并已集成到PyTorch 中，hcu的torch正是通过本算子库进行的集成，方便用户调用，当然用户也可以直接调用本库进行计算。Flash Attention通过减少内存消耗和提高运行速度，使得处理长序列的任务变得更加高效，特别适用于自然语言处理和生物信息学等领域。

本库基于HCU处理器的片上内存和缓存大小，以及数据搬运通路，使用hip算子编程语言优化实现FlashAttention融合算子，充分利用cache、lds及tensor core，提升Attention处理性能。根据实测，在一些场景中FlashAttention算子相比native取得了5倍到10倍以上的性能提升，用户可直接调用相关算子API接口使能大模型极致性能优化。

- **整体计算流程**

$$
\bold{O} = softmax( \frac{\bold{Q}\bold{K^T}}{\sqrt{d_k}}) \bold{V}
$$

![flash-attention-chunk-way.png](../assets/flashattn_banner.jpg)

在不访问整个输入的情况下优化attention计算，并减少相关计算量。重构attention计算，将输入分割成块，并对分块进行多次传递，从而逐步执行attention计算（该步骤称为tiling）。FlashAttention 使用tiling来防止在相对较慢的 GPU显存上实现大型 𝑁 × 𝑁 注意力矩阵（虚线框）计算。在外部循环（红色箭头）中，FlashAttention 循环遍历 K 和 V 矩阵块，并将它们加载到快速片上 SRAM。在每个块中，FlashAttention 循环遍历 Q 矩阵块，将它们加载到 SRAM，并将注意力计算的输出写回 HBM。将输入Q、K、V矩阵分成很多块，将它们从较慢的HBM加载到较快的SRAM，然后在SRAM计算关于这些块的注意力输出。对每个块的计算结果缩放之后进行add操作，则得到正确的结果

## 接口列表

|**算子**|**接口类型**|**推理\(前向\)**|**训练\(反向\)**|
|---|---|---|---|
|flash\_attn\_func|python|支持|支持|
|flash\_attn\_kvpacked\_func|python|支持|支持|
|flash\_attn\_qkvpacked\_func|python|支持|支持|
|flash\_attn\_varlen\_func|python|支持|支持，FP8不支持反向|
|flash\_attn\_varlen\_kvpacked\_func|python|支持|支持|
|flash\_attn\_varlen\_qkvpacked\_func|python|支持|支持|
|vllm\_flash\_attn\_varlen\_func|python|支持|不支持|
|vllm\_flash\_attn\_with\_kvcache|python|支持|不支持|
|flash\_attn\_with\_kvcache|python|支持|不支持|
|flash\_attn\_with\_mask\_func|python|支持|支持，headdim=32仅支持前向|
|flash\_attn\_varlen\_with\_mask\_func|python|支持|支持，headdim=32仅支持前向|
|sparse\_attn\_func|python|支持|不支持|
|sparse\_attn\_varlen\_func|python|支持|不支持|
|spas\_fa2\_attn\_meansim\_cuda|python|支持|不支持|
|spas\_fa2\_attn\_meansim\_topk\_cuda|python|支持|不支持|
|spas\_fa2\_attn\_meansim\_varlen\_cuda|python|支持|不支持|
|spas\_fa2\_attn\_meansim\_topk\_varlen\_cuda|python|支持|不支持|
|sparse\_attn\_with\_sla|python|支持|不支持|
|varlen\_fwd\_unified|python|支持|不支持|

### flash_attn_func

对应的pack版本算子:

- `flash_attn_kvpacked_func`
- `flash_attn_qkvpacked_func`

#### 参数说明

|参数名|类型|形状|说明|
|-|-|-|-|
|q|torch.Tensor|bshd: `(B, Sq, H, D)`；bhsd: `(B, H, Sq, D)`|Query张量|
|k|torch.Tensor|bshd: `(B, Sk, Hk, D)`；bhsd: `(B, Hk, Sk, D)`|Key张量，支持MQA/GQA|
|v|torch.Tensor|bshd: `(B, Sk, Hk, Dv)`；bhsd: `(B, Hk, Sk, Dv)`|Value张量，通常`Dv == D`|
|kv|torch.Tensor|bshd: `(B, Sk, 2, Hk, D)`；bhsd: `(B, Hk, 2, Sk, D)`|`flash_attn_kvpacked_func`使用的K/V packed输入|
|qkv|torch.Tensor|bshd: `(B, S, 3, H, D)`；bhsd: `(B, H, 3, S, D)`|`flash_attn_qkvpacked_func`使用的Q/K/V packed输入|
|dropout_p|float||dropout概率，推理时建议设为0.0|
|softmax_scale|float||QK计算后的缩放值，默认`1 / sqrt(D)`|
|causal|bool||是否使用causal mask|
|window_size|tuple|`(-1, -1)`|滑动窗口大小，`(-1, -1)`表示关闭滑窗|
|softcap|float||softcap参数，`0.0`表示关闭|
|alibi_slopes|torch.Tensor|`(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool||是否使用确定性反向实现|
|return_attn_probs|bool||是否返回`softmax_lse`和`S_dmask`，主要用于测试|
|bhsd|bool||输入是否为`(B, H, S, D)` layout|
|s_aux|torch.Tensor|`(H,)`|Attention Sinks辅助参数，仅`flash_attn_func`支持|

#### 约束与限制

- **输入输出形状**
  - `q`、`k`、`v`需要位于同一设备，且dtype一致。
  - 默认layout为`bshd`，即`(B, S, H, D)`；`bhsd=True`时使用`(B, H, S, D)`。
  - `flash_attn_kvpacked_func`支持MQA/GQA，要求`H % Hk == 0`。
  - `flash_attn_qkvpacked_func`的Q/K/V在同一个tensor中，Q/K/V head数相同，不支持通过更少KV head实现MQA/GQA。
  - 输出张量的batch、sequence、head layout与`q`一致，最后一维为`Dv`。

- **dtype与设备**
  - 支持`torch.float16`和`torch.bfloat16`。
  - 不支持CPU运算。

- **head维度**
  - 前向支持`D <= 512`、`Dv <= 512`。
  - 反向要求`D`为8的倍数，且`D <= 512`。

- **功能限制**
  - `softcap > 0`时不能同时使用`dropout_p > 0`。
  - `return_attn_probs=True`时返回`(out, softmax_lse, S_dmask)`；`S_dmask`只在dropout路径下有有效内容，主要用于测试。
  - `s_aux`要求dtype与Q/K/V一致，仅支持fp16/bf16，形状为`(H,)`，当前最多支持64个head。

#### 调用示例

```Python
import torch
from flash_attn import flash_attn_func

batch_size, seqlen_q, seqlen_k, nheads, nheads_k, d = 4, 1024, 1024, 8, 2, 128
q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=torch.float16)
k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=torch.float16)
v = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v, causal=True)
```

### flash_attn_varlen_func

对应的pack版本算子:

- `flash_attn_varlen_kvpacked_func`
- `flash_attn_varlen_qkvpacked_func`

#### 参数说明

|参数名|类型|形状|说明|
|-|-|-|-|
|q|torch.Tensor|`(total_q, H, D)`|Query张量，`total_q`为batch内所有query token数之和|
|k|torch.Tensor|非paged: `(total_k, Hk, D)`；paged: `(num_blocks, block_size, Hk, D)`|Key张量|
|v|torch.Tensor|非paged: `(total_k, Hk, Dv)`；paged: `(num_blocks, block_size, Hk, Dv)`|Value张量|
|kv|torch.Tensor|`(total_k, 2, Hk, D)`|`flash_attn_varlen_kvpacked_func`使用的K/V packed输入|
|qkv|torch.Tensor|`(total, 3, H, D)`|`flash_attn_varlen_qkvpacked_func`使用的Q/K/V packed输入|
|cu_seqlens_q|torch.Tensor|`(B + 1,)`|query的累积序列长度，dtype为`torch.int32`|
|cu_seqlens_k|torch.Tensor|`(B + 1,)`|key/value的累积序列长度，dtype为`torch.int32`|
|max_seqlen_q|int||batch内最大的query序列长度|
|max_seqlen_k|int||batch内最大的key序列长度|
|dropout_p|float||dropout概率，推理时建议设为0.0|
|softmax_scale|float||QK计算后的缩放值，默认`1 / sqrt(D)`|
|causal|bool||是否使用causal mask|
|window_size|tuple|`(-1, -1)`|滑动窗口大小|
|softcap|float||softcap参数，`0.0`表示关闭|
|alibi_slopes|torch.Tensor|`(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool||是否使用确定性反向实现|
|return_attn_probs|bool||是否返回`softmax_lse`和`S_dmask`，主要用于测试|
|q_descale|torch.Tensor|`(B, Hk)`|FP8 Q反量化scale，仅`flash_attn_varlen_func`支持|
|k_descale|torch.Tensor|`(B, Hk)`|FP8 K反量化scale，仅`flash_attn_varlen_func`支持|
|v_descale|torch.Tensor|`(B, Hk)`|FP8 V反量化scale，仅`flash_attn_varlen_func`支持|
|block_table|torch.Tensor|`(B, max_blocks)`|paged KV cache block索引表，仅`flash_attn_varlen_func`支持|
|s_aux|torch.Tensor|`(H,)`|Attention Sinks辅助参数，仅`flash_attn_varlen_func`支持|

#### 约束与限制

- **输入输出形状**
  - `total_q`是所有batch的`seqlen_q`相加后的值。
  - `total_k`是所有batch的`seqlen_k`相加后的值。
  - `cu_seqlens_q[i + 1] - cu_seqlens_q[i]`表示第`i`个样本的真实query长度。
  - `cu_seqlens_k[i + 1] - cu_seqlens_k[i]`表示第`i`个样本的真实key/value长度。
  - `cu_seqlens_q`和`cu_seqlens_k`必须为设备侧`torch.int32` contiguous tensor。
  - `block_table`使用时，K/V采用paged layout: `(num_blocks, block_size, Hk, D/Dv)`。

- **dtype与设备**
  - 常规前向和反向支持`torch.float16`、`torch.bfloat16`。
  - `flash_attn_varlen_func`前向支持`torch.float8_e4m3fn`，需要提供`q_descale/k_descale/v_descale`。
  - FP8路径不支持反向。
  - 不支持CPU运算。

- **head维度**
  - 主要变长路径支持`D <= 256`、`Dv <= 256`。
  - 反向只支持fp16/bf16，且不支持FP8。

- **功能限制**
  - `flash_attn_varlen_kvpacked_func`和`flash_attn_varlen_qkvpacked_func`不暴露`q_descale/k_descale/v_descale`、`block_table`、`s_aux`参数。
  - `softcap > 0`时不能同时使用`dropout_p > 0`。
  - paged KV的`block_size`在gfx936/gfx938上要求为64的倍数，在gfx928上要求为16的倍数。
  - `return_attn_probs=True`时返回`(out, softmax_lse, S_dmask)`；`S_dmask`主要用于测试。

#### 调用示例

```Python
import torch
from flash_attn import flash_attn_varlen_func

lengths_q = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
lengths_k = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
cu_q = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), lengths_q.cumsum(0)])
cu_k = torch.cat([torch.zeros(1, device="cuda", dtype=torch.int32), lengths_k.cumsum(0)])

q = torch.randn(int(lengths_q.sum()), 8, 128, device="cuda", dtype=torch.float16)
k = torch.randn(int(lengths_k.sum()), 2, 128, device="cuda", dtype=torch.float16)
v = torch.randn(int(lengths_k.sum()), 2, 128, device="cuda", dtype=torch.float16)

out = flash_attn_varlen_func(
    q, k, v, cu_q, cu_k,
    max_seqlen_q=int(lengths_q.max()),
    max_seqlen_k=int(lengths_k.max()),
    causal=True,
)
```

### vllm\_flash\_attn\_varlen\_func

#### 参数说明

|参数名|类型 / 形状|说明|备注|
|-|-|-|-|
|q|Tensor `(total_q, H, D)`|Query张量||
|k|Tensor|Key张量或paged Key cache|prefix cache路径使用vLLM paged layout|
|v|Tensor|Value张量或paged Value cache|prefix cache路径使用vLLM paged layout|
|max_seqlen_q|int|batch内最大的query序列长度||
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query的累积序列长度||
|max_seqlen_k|int|batch内最大的key序列长度||
|cu_seqlens_k|Tensor `(B + 1,)`, int32|普通varlen路径使用的key/value累积序列长度|与`seqused_k`二选一|
|seqused_k|Tensor `(B,)`, int32|每个样本实际使用的KV token数|与`cu_seqlens_k`二选一|
|q_v|Tensor|兼容参数|当前Python接口不使用该参数|
|dropout_p|float|attention dropout概率||
|softmax_scale|float|QK计算后的缩放系数|默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask||
|window_size|List[int]或None|滑窗attention `[left, right]`|None表示`(-1, -1)`|
|softcap|float|softcap参数||
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置||
|deterministic|bool|兼容参数|该接口不支持反向|
|return_attn_probs|bool|兼容参数|当前不返回attention probabilities|
|block_table|Tensor `(B, max_blocks)`, int32|paged KV cache block索引表|使用paged KV时需要|
|return_softmax_lse|bool|是否返回softmax LSE|部分快速decode场景只返回out|
|out|Tensor|预分配输出buffer||
|is_prefix_cache|bool|是否启用prefix cache / paged attention||
|scheduler_metadata|Any|兼容参数|当前Python接口不使用该参数|
|q_descale|Tensor|FP8 Q反量化scale|FP8路径使用|
|k_descale|Tensor|FP8 K反量化scale|FP8路径使用|
|v_descale|Tensor|FP8 V反量化scale|FP8路径使用|
|kv_cache_dtype|str|KV cache数据类型|decode快速路径使用|
|fa_version|int|FlashAttention版本选择|仅支持2|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数||
|is_bhsd|bool|paged attention兼容参数||

#### 约束与限制

- **输入与layout**
  - 必须且只能提供`cu_seqlens_k`或`seqused_k`之一。
  - `block_table`非None时必须提供`seqused_k`。
  - `fa_version`仅支持`2`。
  - `is_prefix_cache=True`时，K/V使用vLLM paged layout:
    - `k`: `(num_blocks, Hk, block_size, D)`
    - `v`: `(num_blocks, Hk, Dv, block_size)`
  - `is_prefix_cache=False`时，普通varlen K/V layout与`flash_attn_varlen_func`一致；paged K/V使用`(num_blocks, block_size, Hk, D/Dv)`。

- **dtype与设备**
  - 非FP8路径支持fp16/bf16。
  - FP8 prefix cache路径有条件支持`float8_e4m3fn`或`float8_e5m2`。
  - 当K/V为`float8_e5m2`且Q不是FP8时，Q应为fp16/bf16。
  - 当Q为FP8且未传入`out`时，输出默认使用bf16。
  - 不支持CPU运算。

- **head维度**
  - `is_prefix_cache=True`时，`D`必须是64的倍数且`D <= 256`。

- **功能限制**
  - 仅支持前向，不支持反向。
  - `q_v`、`scheduler_metadata`、`deterministic`、`return_attn_probs`为兼容参数，当前不改变实际计算行为。
  - 部分快速decode场景即使`return_softmax_lse=True`也只返回`out`。
  - `s_aux`要求dtype为fp16/bf16且与Q/K/V匹配，不应与Q为FP8的路径混用。

### vllm\_flash\_attn\_with\_kvcache

#### 参数说明

|参数名|类型 / 形状|说明|
|-|-|-|
|q|Tensor `(B, Sq, H, D)`|Query张量|
|k_cache|Tensor|Key cache。vLLM paged layout为`(num_blocks, Hk, block_size, D)`|
|v_cache|Tensor|Value cache。vLLM paged layout为`(num_blocks, Hk, Dv, block_size)`|
|k|Tensor `(B, S_new, Hk, D)`|新生成的Key，可选|
|v|Tensor `(B, S_new, Hk, Dv)`|新生成的Value，可选|
|rotary_cos|Tensor `(S_ro, rotary_dim / 2)`|Rotary embedding cos表|
|rotary_sin|Tensor `(S_ro, rotary_dim / 2)`|Rotary embedding sin表|
|cache_seqlens|int或Tensor `(B,)`|每个batch当前KV cache长度，Tensor dtype为int32|
|cache_batch_idx|Tensor `(B,)`|映射到cache的batch index，dtype为int32|
|cache_leftpad|Tensor `(B,)`|KV cache起始偏移，dtype为int32|
|block_table|Tensor `(B, max_blocks)`|Paged KV cache block索引表，dtype为int32|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|window_size|tuple|Sliding window attention|
|softcap|float|softcap参数|
|rotary_interleaved|bool|Rotary embedding排列方式|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi bias|
|num_splits|int|KV sequence切分数|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数|
|return_softmax_lse|bool|是否返回softmax LSE|
|q_scale|Tensor|FP8 Q scale|
|k_scale|Tensor|FP8 K scale|
|v_scale|Tensor|FP8 V scale|
|kv_cache_dtype|str|KV cache dtype|
|out|Tensor|输出buffer|
|max_seqlen_k|int|KV cache最大长度|
|is_bhsd|bool|paged attention兼容参数|

#### 约束与限制

- **KV cache dtype与layout**
  - vLLM paged layout中，`k_cache`为`(num_blocks, Hk, block_size, D)`，`v_cache`为`(num_blocks, Hk, Dv, block_size)`。
  - 快速decode场景可通过`kv_cache_dtype`和`q_scale/k_scale/v_scale`接入FP8 cache。
  - 非快速decode路径要求`k_cache`为fp16或bf16。

- **head维度**
  - `D`必须是64的倍数且`D <= 256`。
  - Q head数必须能被KV head数整除。

- **功能限制**
  - 仅支持前向，不支持反向。
  - `k`和`v`需要同时传入或同时为None。
  - 传入`k/v`时会原地更新KV cache，调用方需要保证cache空间足够。
  - 快速decode场景不返回`softmax_lse`，即使`return_softmax_lse=True`也只返回`out`。
  - `rotary_cos`和`rotary_sin`必须同时传入，rotary dim不能超过`D`且需要能被16整除。
  - fp16/bf16有少量seq_len非对齐下的精度错误，详见单测test_prefix_kvcache.py。预计下个版本解决。

### flash\_attn\_with\_kvcache

#### 参数说明

|参数名|类型 / 形状|说明|
|-|-|-|
|q|Tensor `(B, Sq, H, D)`|Query张量|
|k_cache|Tensor|非paged: `(B_cache, Sk_cache, Hk, D)`；paged: `(num_blocks, block_size, Hk, D)`|
|v_cache|Tensor|非paged: `(B_cache, Sk_cache, Hk, Dv)`；paged: `(num_blocks, block_size, Hk, Dv)`|
|k|Tensor `(B, S_new, Hk, D)`|新生成的Key，可选|
|v|Tensor `(B, S_new, Hk, Dv)`|新生成的Value，可选|
|rotary_cos|Tensor `(S_ro, rotary_dim / 2)`|Rotary embedding cos表|
|rotary_sin|Tensor `(S_ro, rotary_dim / 2)`|Rotary embedding sin表|
|cache_seqlens|int或Tensor `(B,)`|每个batch当前KV cache长度，Tensor dtype为int32|
|cache_batch_idx|Tensor `(B,)`|映射到cache的batch index，dtype为int32|
|cache_leftpad|Tensor `(B,)`|KV cache起始偏移，dtype为int32|
|block_table|Tensor `(B, max_blocks)`|Paged KV cache block索引表，dtype为int32|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|window_size|tuple|Sliding window attention|
|softcap|float|softcap参数|
|rotary_interleaved|bool|Rotary embedding排列方式|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi bias|
|num_splits|int|KV sequence切分数|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数|
|return_softmax_lse|bool|是否返回softmax LSE|
|out|Tensor|输出buffer|

#### 约束与限制

- **KV cache dtype与layout**
  - `q`、`k_cache`、`v_cache`支持fp16/bf16，且dtype需要一致。
  - `k_cache`和`v_cache`最后一维必须contiguous。
  - paged KV cache layout为`(num_blocks, block_size, Hk, D/Dv)`。

- **KV cache更新**
  - `k`和`v`需要同时传入或同时为None。
  - 传入`k/v`时会原地更新`k_cache/v_cache`，调用方需要保证cache空间足够。
  - paged KV cache不支持同时使用`cache_batch_idx`。
  - `cache_leftpad`不支持paged KV cache。

- **head维度与功能**
  - 常规KV cache路径支持`D <= 256`。
  - Q head数必须能被KV head数整除。
  - `rotary_cos`和`rotary_sin`必须同时传入，rotary dim不能超过`D`且需要能被16整除。
  - 仅支持前向，不支持反向。

### flash\_attn\_with\_mask\_func

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor，bshd: `(B, Sq, H, D)`；bhsd: `(B, H, Sq, D)`|Query张量|
|k|Tensor，bshd: `(B, Sk, Hk, D)`；bhsd: `(B, Hk, Sk, D)`|Key张量|
|v|Tensor，bshd: `(B, Sk, Hk, Dv)`；bhsd: `(B, Hk, Sk, Dv)`|Value张量|
|attn_mask|Tensor `(B, H, Sq, Sk)`|显式attention mask|
|dropout_p|float|attention dropout概率|
|softmax_scale|float|QK计算后的缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|window_size|tuple|滑窗attention|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|是否使用确定性反向实现|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|bhsd|bool|输入是否为`(B, H, S, D)`格式|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数|

#### 约束与限制

- **mask语义**
  - bool mask: `True`表示attend，`False`表示mask out。
  - integer mask: Python层按`attn_mask != 0`转换，非0表示attend。
  - float mask: Python层按`attn_mask == 0`转换，0表示attend，非0或`-inf`表示mask out。
  - `attn_mask`最后一维必须contiguous。

- **dtype与head维度**
  - Q/K/V仅支持fp16/bf16。
  - 前向支持`D in {32, 64, 128}`。
  - 反向支持`D in {64, 128}`，因此`D=32`仅建议用于forward-only。

- **功能限制**
  - Q head数必须能被KV head数整除。
  - causal mask、sliding window和显式mask共同生效。
  - 全mask行输出为0。
  - `softcap > 0`时不能同时使用`dropout_p > 0`。

### flash\_attn\_varlen\_with\_mask\_func

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(total_q, H, D)`|Query张量|
|k|Tensor `(total_k, Hk, D)`|Key张量|
|v|Tensor `(total_k, Hk, Dv)`|Value张量|
|attn_mask|Tensor `(B, H, max_seqlen_q, max_seqlen_k)`|显式attention mask|
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query累积序列长度|
|cu_seqlens_k|Tensor `(B + 1,)`, int32|key/value累积序列长度|
|max_seqlen_q|int|batch内最大的query序列长度|
|max_seqlen_k|int|batch内最大的key序列长度|
|dropout_p|float|attention dropout概率|
|softmax_scale|float|QK计算后的缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|window_size|tuple|滑窗attention|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|是否使用确定性反向实现|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|zero_tensors|bool|是否在计算前清零中间输出|
|seqused_k|Tensor `(B,)`, int32|每个样本实际使用的KV token数，可选|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数|

#### 约束与限制

- **mask语义**
  - bool mask: `True`表示attend，`False`表示mask out。
  - integer mask: Python层按`attn_mask != 0`转换，非0表示attend。
  - float mask: Python层按`attn_mask == 0`转换，0表示attend，非0或`-inf`表示mask out。
  - `attn_mask`最后一维必须contiguous。

- **dtype与head维度**
  - Q/K/V仅支持fp16/bf16。
  - 前向支持`D in {32, 64, 128}`。
  - 反向支持`D in {64, 128}`，因此`D=32`仅建议用于forward-only。

- **功能限制**
  - `cu_seqlens_q`、`cu_seqlens_k`必须为设备侧`torch.int32` contiguous tensor。
  - `attn_mask`按`max_seqlen_q/max_seqlen_k`组织，超出真实序列长度的区域不参与有效计算。
  - Q head数必须能被KV head数整除。
  - causal mask、sliding window和显式mask共同生效。
  - `softcap > 0`时不能同时使用`dropout_p > 0`。

### sparse\_attn\_func

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(B, Sq, H, D)`|Query张量|
|k|Tensor `(B, Sk, Hk, D)`|Key张量|
|v|Tensor `(B, Sk, Hk, Dv)`|Value张量|
|block_count|Tensor `(B, H, num_q_blocks)`|每个query block保留的slash block数量|
|block_offset|Tensor `(B, H, num_q_blocks, NNZ_S)`|slash sparse的K block row offset|
|column_count|Tensor `(B, H, num_q_blocks)`|vertical sparse保留列数量|
|column_index|Tensor `(B, H, num_q_blocks, NNZ_V)`|vertical sparse列索引|
|dropout_p|float|attention dropout概率|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|return_softmax_lse|bool|是否返回softmax LSE|
|out|Tensor|预分配输出buffer|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|
|is_sla|bool|是否使用SLA sparse路径|

#### 约束与限制

- **dtype与head维度**
  - 常规sparse用户路径建议使用fp16/bf16。
  - 非SLA sparse算子当前仅支持`D=128`。
  - `is_sla=True`时，fp16/bf16支持`D=64`或`D=128`，FP8仅支持`D=128`。

- **sparse map**
  - `block_offset`记录row offset，不是原始block id。
  - `num_q_blocks`通常为`ceil(Sq / BLOCK_M)`；当前常用`BLOCK_M=64`，长序列场景可能使用128。

- **功能限制**
  - 仅支持前向，不支持反向。
  - Q head数必须能被KV head数整除。

### sparse\_attn\_varlen\_func

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(total_q, H, D)`|Query张量|
|k|Tensor `(total_k, Hk, D)`|Key张量|
|v|Tensor `(total_k, Hk, Dv)`|Value张量|
|block_count|Tensor `(B, H, max_q_blocks)`|每个query block保留的slash block数量|
|block_offset|Tensor `(B, H, max_q_blocks, NNZ_S)`|slash sparse的K block row offset|
|column_count|Tensor `(B, H, max_q_blocks)`|vertical sparse保留列数量|
|column_index|Tensor `(B, H, max_q_blocks, NNZ_V)`|vertical sparse列索引|
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query累积序列长度|
|cu_seqlens_k|Tensor `(B + 1,)`, int32|key/value累积序列长度|
|max_seqlen_q|int|batch内最大的query序列长度|
|max_seqlen_k|int|batch内最大的key序列长度|
|dropout_p|float|attention dropout概率|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|return_softmax_lse|bool|是否返回softmax LSE|
|out|Tensor|预分配输出buffer|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|

#### 约束与限制

- **dtype与head维度**
  - 支持fp16/bf16。
  - 当前非SLA sparse算子仅支持`D=128`。

- **输入限制**
  - `cu_seqlens_q`、`cu_seqlens_k`必须为设备侧`torch.int32` contiguous tensor。
  - Python接口未暴露`seqused_k`，按`cu_seqlens_k`描述KV长度。

- **功能限制**
  - 仅支持前向，不支持反向。
  - Q head数必须能被KV head数整除。

### spas\_fa2\_attn\_meansim\_cuda

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(B, Sq, H, D)`|Query张量|
|k|Tensor `(B, Sk, Hk, D)`|Key张量|
|v|Tensor `(B, Sk, Hk, Dv)`|Value张量|
|dropout_p|float|attention dropout概率，默认0.0|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|smooth_k|bool|生成稀疏结构时是否对K做减均值平滑|
|simthreshd1|float|mean-sim相似度阈值|
|cdfthreshd|float|CDF分位阈值|
|attention_sink|bool|是否保留首块作为attention sink|
|block_m|int|Q维度块大小，默认64|
|block_k|int|K维度块大小，默认64|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|
|return_softmax_lse|bool|是否返回softmax LSE|
|return_sparsity|bool|是否返回稀疏率|
|out|Tensor|预分配输出buffer|

#### 约束与限制

- **输入限制**
  - 输入为定长layout `(B, S, H, D)`。
  - `q.dim()`必须为4。
  - `Sq >= 128`。
  - `block_k`必须等于`block_m`。

- **dtype与head维度**
  - 输入为float32或float16时，内部转为float16计算。
  - 其他输入dtype会内部转为bfloat16计算。
  - 非SLA sparse算子当前仅支持`D=128`。

- **功能限制**
  - 仅支持前向，不支持反向。
  - `smooth_k`仅影响稀疏结构生成，不表示底层attention计算使用了平滑后的K。
  - `return_sparsity=True`时返回`(out, sparsity)`；若同时`return_softmax_lse=True`，返回`(out, softmax_lse, sparsity)`。

### spas\_fa2\_attn\_meansim\_topk\_cuda

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(B, Sq, H, D)`|Query张量|
|k|Tensor `(B, Sk, Hk, D)`|Key张量|
|v|Tensor `(B, Sk, Hk, Dv)`|Value张量|
|dropout_p|float|attention dropout概率，默认0.0|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|smooth_k|bool|生成稀疏结构时是否对K做减均值平滑|
|simthreshd1|float|mean-sim相似度阈值，默认-0.1|
|topk|float|每个query块保留的key块比例，默认0.5|
|attention_sink|bool|是否保留首块作为attention sink|
|block_m|int|Q维度块大小，默认64|
|block_k|int|K维度块大小，默认64|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|
|return_softmax_lse|bool|是否返回softmax LSE|
|return_sparsity|bool|是否返回稀疏率|
|out|Tensor|预分配输出buffer|

#### 约束与限制

- 约束同`spas_fa2_attn_meansim_cuda`。
- 稀疏结构由`topk`和mean-sim结果共同决定。
- 仅支持前向，不支持反向。

### spas\_fa2\_attn\_meansim\_varlen\_cuda

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(total_q, H, D)`|Query张量|
|k|Tensor `(total_k, Hk, D)`|Key张量|
|v|Tensor `(total_k, Hk, Dv)`|Value张量|
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query累积序列长度|
|cu_seqlens_k|Tensor `(B + 1,)`, int32|key/value累积序列长度|
|max_seqlen_q|int|batch内最大query序列长度|
|max_seqlen_k|int|batch内最大key序列长度|
|dropout_p|float|attention dropout概率，默认0.0|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|smooth_k|bool|生成稀疏结构时是否对K做减均值平滑|
|simthreshd1|float|mean-sim相似度阈值|
|cdfthreshd|float|CDF分位阈值|
|attention_sink|bool|是否保留首块作为attention sink|
|block_m|int|Q维度块大小，默认64|
|block_k|int|K维度块大小，默认64|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|
|return_softmax_lse|bool|是否返回softmax LSE|
|return_sparsity|bool|是否返回稀疏率|
|out|Tensor|预分配输出buffer|

#### 约束与限制

- **输入限制**
  - 输入为变长layout `(total, H, D)`。
  - `q.dim()`、`k.dim()`、`v.dim()`必须为3。
  - `cu_seqlens_q`和`cu_seqlens_k`描述的batch数必须一致。
  - `block_m == block_k == 64`。

- **dtype与head维度**
  - 支持fp16/bf16。
  - 非SLA sparse算子当前仅支持`D=128`。

- **功能限制**
  - 仅支持前向，不支持反向。
  - `smooth_k`仅影响稀疏结构生成。
  - `return_sparsity=True`时返回`(out, sparsity)`；若同时`return_softmax_lse=True`，返回`(out, softmax_lse, sparsity)`。

### spas\_fa2\_attn\_meansim\_topk\_varlen\_cuda

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(total_q, H, D)`|Query张量|
|k|Tensor `(total_k, Hk, D)`|Key张量|
|v|Tensor `(total_k, Hk, Dv)`|Value张量|
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query累积序列长度|
|cu_seqlens_k|Tensor `(B + 1,)`, int32|key/value累积序列长度|
|max_seqlen_q|int|batch内最大query序列长度|
|max_seqlen_k|int|batch内最大key序列长度|
|dropout_p|float|attention dropout概率，默认0.0|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|deterministic|bool|兼容参数，该接口不支持反向|
|return_attn_probs|bool|是否返回attention prob，主要用于测试|
|smooth_k|bool|生成稀疏结构时是否对K做减均值平滑|
|simthreshd1|float|mean-sim相似度阈值，默认-0.1|
|topk|float|每个query块保留的key块比例，默认0.5|
|attention_sink|bool|是否保留首块作为attention sink|
|block_m|int|Q维度块大小，默认64|
|block_k|int|K维度块大小，默认64|
|pv_threshold|float|动态跳过P@V的阈值|
|enable_dynamic_skip|bool|是否开启动态PV跳过优化|
|return_softmax_lse|bool|是否返回softmax LSE|
|return_sparsity|bool|是否返回稀疏率|
|out|Tensor|预分配输出buffer|

#### 约束与限制

- 约束同`spas_fa2_attn_meansim_varlen_cuda`。
- 稀疏结构由`topk`和mean-sim结果共同决定。
- 仅支持前向，不支持反向。

### sparse\_attn\_with\_sla

#### 参数说明

|参数名|类型|形状|说明|
|-|-|-|-|
|q|torch.Tensor|`(B, S, H, D)`|Query张量|
|k|torch.Tensor|`(B, S, Hk, D)`|Key张量|
|v|torch.Tensor|`(B, S, Hk, Dv)`|Value张量|
|topk|float||key block选取比例|
|feature_map|str||线性attention激活函数，可选`elu`、`relu`、`softmax`|
|use_bf16|bool||是否使用bf16|
|use_fp8|bool||是否使用fp8|
|return_sparsity|bool||是否返回稀疏率|

#### 约束与限制

- **dtype与head维度**
  - `use_bf16`和`use_fp8`不能同时为True。
  - fp16/bf16支持`D=64`或`D=128`。
  - FP8仅支持`D=128`。

- **功能限制**
  - 输入使用定长layout `(B, S, H, D)`。
  - Q head数必须能被KV head数整除。
  - 仅支持前向，不支持反向。
  - `return_sparsity=True`时返回`(out, sparsity)`。

### varlen\_fwd\_unified

#### 参数说明

|参数名|类型 / 形状|说明|
|---|---|---|
|q|Tensor `(total_q, H, D)`|Query张量|
|k|Tensor|paged Key cache，layout由`layout`决定|
|v|Tensor|paged Value cache，layout由`layout`决定|
|cu_seqlens_q|Tensor `(B + 1,)`, int32|query累积序列长度|
|seqused_k|Tensor `(B,)`, int32|每个样本实际使用的KV token数|
|block_table|Tensor `(B, max_blocks)`, int32|paged KV cache block索引表|
|max_seqlen_q|int|batch内最大query序列长度|
|max_seqlen_k|int|batch内最大key序列长度|
|softmax_scale|float|QK缩放系数，默认`1 / sqrt(D)`|
|causal|bool|是否使用causal mask|
|softcap|float|softcap参数|
|window_size|tuple|滑窗attention|
|alibi_slopes|Tensor `(H,)`或`(B, H)`|ALiBi位置偏置|
|use_alibi_sqrt|bool|ALiBi兼容参数|
|qq_bias|Tensor|可选bias|
|s_aux|Tensor `(H,)`|Attention Sinks辅助参数|
|mm_prefix_range|Tensor|可选prefix范围参数|
|layout|str|`bshd`或`bhsd`|
|out|Tensor|预分配输出buffer|
|return_softmax_lse|bool|是否返回softmax LSE|
|q_descale|Tensor|FP8 Q反量化scale，可选|
|k_descale|Tensor|FP8 K反量化scale，可选|
|v_descale|Tensor|FP8 V反量化scale，可选|

#### 约束与限制

- **layout**
  - `layout`仅支持`bshd`或`bhsd`。
  - `layout="bshd"`时，K/V形状为`(num_blocks, block_size, Hk, D/Dv)`。
  - `layout="bhsd"`时，K形状为`(num_blocks, Hk, block_size, D)`，V形状为`(num_blocks, Hk, block_size, Dv)`。
  - 注意：`layout="bhsd"`的V layout不同于vLLM paged V layout。

- **dtype与head维度**
  - 支持fp16/bf16。
  - 支持FP8，需要提供对应descale。
  - 支持`D=128`、`Dv=128`、`D=192`、`Dv=128`、`D=256`、`Dv=256`、`D=512`、`Dv=512`等场景。

- **功能限制**
  - 仅支持前向，不支持反向。
  - `cu_seqlens_q`、`seqused_k`、`block_table`必须为设备侧`torch.int32` contiguous tensor。
  - gfx936/gfx938要求page block size为64的倍数。

## Block table \& kv cache shape

**Block table** shape = [batch\_size, max\_blocks\_per\_sequence]

**Block\_size**：单个 block 对应的序列长度，一般默认 64

构造流程如下：

- 对第i个seq进行`blocks_i = ceil(seqlen_i / block_size)`，可以得到第i个seq所需要的block数

- num\_total\_blocks 表示总block数，`num_total_blocks = sum(blocks_i for all i)`

- max\_blocks\_per\_sequence 表示单个seq需要的最大block数，用于分配 block\_table 容量，`max\_blocks\_per\_sequence=max(blocks_i)`

- block\_table可以构造为`[batch_size, max_blocks_per_sequence]`

`block_table[b, i]` = 第 b 个序列的第 i 个 block 在 kv\_cache 里的索引，若某序列 block 数 \< max\_blocks\_per\_sequence，其余位置填 -1

**k v 形状设置:**

  1. flash\_attn\_varlen\_func，flash\_attn\_with\_kvcache接口中
    k shape: `[num_total_blocks, block_size, h, d]`
    v shape: `[num_total_blocks, block_size, h, d]`
  2. vllm\_flash\_attn\_varlen\_func，vllm\_flash\_attn\_with\_kvcache接口中
    k shape: `[num_total_blocks, h, block_size, d]`
    v shape: `[num_total_blocks, h, d, block_size]`
  3. varlen\_fwd\_unified接口中
    - `layout="bshd"`: k/v shape 为 `[num_total_blocks, block_size, h, d]`
    - `layout="bhsd"`: k shape 为 `[num_total_blocks, h, block_size, d]`，v shape 为 `[num_total_blocks, h, block_size, d]`

可参考随机生成代码：

```Python
...
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )
...

def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks
```
