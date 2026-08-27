import pytest
import math
import random

import torch
import triton

from flash_attn import vllm_flash_attn_varlen_func
torch.set_printoptions(precision=4, profile="default", sci_mode=False)

# ----------------- scaled dot-product -----------------
def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    tmp = query @ key.transpose(-2, -1)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse

def scaled_dot_product_attention_int8(query, key, value, h_q, h_kv, k_scale, v_scale, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    for i in range(key.shape[0]):
        key[i] = key[i] * k_scale[i]
        value[i] = value[i] * v_scale[i]
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight_temp = query @ key.transpose(-2, -1)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse

# ----------------- 对比函数 -----------------
def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False, is_e5m2: bool = False) -> None:
    torch_dtype = x.dtype
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if is_e5m2:
        assert cos_diff < 1e-2
    elif use_fp8:
        assert cos_diff < 1e-3
    else:
        assert cos_diff < (1e-4 if torch_dtype == torch.bfloat16 else 1e-5)


# ================= test_flash_kvcache (fp16/bf16) =================
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", [64, 128, 192, 256])
@pytest.mark.parametrize("b", [4])
@pytest.mark.parametrize(
    "s_q,mean_sk",
    [
        (16,   128),
        (64,   256),
        (128,  512),
        (256,  512),
        (512,  1024),
        (1024, 2048),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_kvcache(b, s_q, mean_sk, mha_type, d, causal, varlen, dtype):
    if mha_type == "mha":
        h_q, h_kv = 16, 16
    elif mha_type == "mqa":
        h_q, h_kv = 16, 1
    else:  # gqa
        h_q, h_kv = 16, 4

    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    random.seed(42)
    torch.manual_seed(42)

    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {causal=}, {varlen=}, {dtype=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(int(random.normalvariate(mean_sk, mean_sk / 2)), s_q)
    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 64) * 64

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
        blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    blocked_k_ = blocked_k.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v.permute(0, 2, 3, 1).contiguous()

    cu_seqlens_q = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32, device=q.device)
    q_varlen = q.reshape(b * s_q, h_q, d)

    out, lse, _ = vllm_flash_attn_varlen_func(
        q=q_varlen,
        k=blocked_k_,
        v=blocked_v_,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=s_q,
        seqused_k=cache_seqlens,
        max_seqlen_k=max_seqlen,
        block_table=block_table,
        causal=causal,
        return_softmax_lse=True,
    )
    out_flash = out.view(b, s_q, h_q, d)
    lse_flash = lse.view(h_q, b, s_q).permute(1, 0, 2)

    out_torch = torch.empty(b, s_q, h_q, d, dtype=torch.float32)
    lse_torch = torch.empty(b, h_q, s_q, dtype=torch.float32)
    for i in range(b):
        begin = i * max_seqlen_pad
        end = begin + cache_seqlens[i]
        O, LSE = scaled_dot_product_attention(
            q[i].transpose(0, 1),
            blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
            blocked_v.view(-1, h_kv, d)[begin:end].transpose(0, 1),
            h_q=h_q,
            h_kv=h_kv,
            is_causal=causal,
        )
        out_torch[i] = O.transpose(0, 1)
        lse_torch[i] = LSE

    print("Max abs diff out:", (out_flash - out_torch).abs().max().item())
    cal_diff(out_flash, out_torch, "out")


# ================= test_flash_kvcache_fp8 (e4m3/e5m2) =================
@pytest.mark.parametrize("torch_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("d", [64, 128, 192, 256])
@pytest.mark.parametrize("b", [4])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "s_q,mean_sk",
    [
        (16,   128),
        (64,   256),
        (128,  512),
        (256,  512),
        (512,  1024),
        (1024, 2048),
        (2043, 2048),
        (4077, 4096),
    ],
)
@torch.inference_mode()
def test_flash_kvcache_fp8(s_q, mean_sk, b, d, mha_type, causal, varlen, torch_dtype, output_dtype):
    if mha_type == "mha":
        h_q, h_kv = 16, 16
    elif mha_type == "mqa":
        h_q, h_kv = 16, 1
    else:  # gqa
        h_q, h_kv = 16, 4

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.manual_seed(42)
    random.seed(42)

    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {causal=}, {varlen=}, {torch_dtype=}, {output_dtype=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)

    total_seqlens = cache_seqlens.sum().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 64) * 64
    print(f"{total_seqlens=}, {max_seqlen=}, {max_seqlen_pad=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 64
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, d)

    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
        blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

    q_fp8 = q.to(torch_dtype)
    blocked_k_fp8 = blocked_k.to(torch_dtype)
    blocked_v_fp8 = blocked_v.to(torch_dtype)

    q = q_fp8
    blocked_k_ = blocked_k_fp8.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v_fp8.permute(0, 2, 3, 1).contiguous()

    q_descale, k_descale, v_descale = [torch.ones(b, h_kv, dtype=torch.float32) for _ in range(3)]

    cu_seqlens_q = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32, device=q.device)
    q_varlen = q.reshape(b * s_q, h_q, d)

    output = torch.zeros(b, s_q, h_q, d, dtype=output_dtype, device=q.device)  # 改用 output_dtype

    def flash_kvcache():
        out, lse, _ = vllm_flash_attn_varlen_func(
            q=q_varlen,
            k=blocked_k_,
            v=blocked_v_,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=s_q,
            seqused_k=cache_seqlens,
            max_seqlen_k=max_seqlen,
            block_table=block_table,
            causal=causal,
            return_softmax_lse=True,
            out=output.view(b * s_q, h_q, d),
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out.view(b, s_q, h_q, d), lse.view(h_q, b, s_q).permute(1, 0, 2)

    def ref_kvcache():
        q_ = q_fp8.to(torch.float)
        blocked_k_ref = blocked_k_fp8.to(torch.float)
        blocked_v_ref = blocked_v_fp8.to(torch.float)
        out = torch.empty(b, s_q, h_q, d, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q_[i].transpose(0, 1),
                blocked_k_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_kvcache()
    out_torch, lse_torch = ref_kvcache()

    cal_diff(out_flash, out_torch, "out", use_fp8=True, is_e5m2=(torch_dtype == torch.float8_e5m2))

# # ================= test_flash_kvcache_kv_fp8 (q: bf16/fp16, k/v: e5m2) =================
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("varlen", [False, True])
# @pytest.mark.parametrize("mha_type", ["mha"])
# # @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize("b", [4])
# @pytest.mark.parametrize(
#     "s_q,mean_sk",
#     [
#         (16,   128),
#         (64,   256),
#         (128,  512),
#         (256,  512),
#         (512,  1024),
#         (1024, 2048),
#         (2043, 2048),
#         (4077, 4096),
#     ],
# )
# @torch.inference_mode()
# def test_flash_kvcache_kv_fp8(s_q, mean_sk, b, d, mha_type, causal, varlen, dtype):
#     if mha_type == "mha":
#         h_q, h_kv = 16, 16
#     elif mha_type == "mqa":
#         h_q, h_kv = 16, 1
#     else:  # gqa
#         h_q, h_kv = 16, 4

#     device = torch.device("cuda:0")
#     torch.set_default_dtype(dtype)
#     torch.set_default_device(device)
#     torch.manual_seed(42)
#     random.seed(42)

#     print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {causal=}, {varlen=}, {dtype=}, kv_dtype=e5m2")

#     cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
#     if varlen:
#         for i in range(b):
#             cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)

#     total_seqlens = cache_seqlens.sum().item()
#     max_seqlen = cache_seqlens.max().item()
#     max_seqlen_pad = triton.cdiv(max_seqlen, 64) * 64
#     print(f"{total_seqlens=}, {max_seqlen=}, {max_seqlen_pad=}")

#     q = torch.randn(b, s_q, h_q, d)
#     block_size = 64
#     block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
#     blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
#     blocked_v = torch.randn(block_table.numel(), block_size, h_kv, d)

#     for i in range(b):
#         blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
#         blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")

#     # q 保持 fp16/bf16，k/v 转成 e5m2
#     blocked_k_fp8 = blocked_k.to(torch.float8_e5m2)
#     blocked_v_fp8 = blocked_v.to(torch.float8_e5m2)

#     blocked_k_ = blocked_k_fp8.permute(0, 2, 1, 3).contiguous()
#     blocked_v_ = blocked_v_fp8.permute(0, 2, 3, 1).contiguous()

#     cu_seqlens_q = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32, device=q.device)
#     q_varlen = q.reshape(b * s_q, h_q, d)

#     # output dtype 与 q 一致
#     output = torch.zeros(b, s_q, h_q, d, dtype=dtype, device=q.device)

#     def flash_kvcache():
#         out, lse, _ = vllm_flash_attn_varlen_func(
#             q=q_varlen,
#             k=blocked_k_,
#             v=blocked_v_,
#             cu_seqlens_q=cu_seqlens_q,
#             max_seqlen_q=s_q,
#             seqused_k=cache_seqlens,
#             max_seqlen_k=max_seqlen,
#             block_table=block_table,
#             causal=causal,
#             return_softmax_lse=True,
#             out=output.view(b * s_q, h_q, d),
#         )
#         return out.view(b, s_q, h_q, d), lse.view(h_q, b, s_q).permute(1, 0, 2)

#     def ref_kvcache():
#         # q 保持原始 fp16/bf16，k/v 从 e5m2 还原
#         blocked_k_ref = blocked_k_fp8.to(torch.float)
#         blocked_v_ref = blocked_v_fp8.to(torch.float)
#         out = torch.empty(b, s_q, h_q, d, dtype=torch.float32)
#         lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
#         for i in range(b):
#             begin = i * max_seqlen_pad
#             end = begin + cache_seqlens[i]
#             O, LSE = scaled_dot_product_attention(
#                 q[i].transpose(0, 1),           # q 直接用原始值
#                 blocked_k_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
#                 blocked_v_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
#                 h_q=h_q,
#                 h_kv=h_kv,
#                 is_causal=causal,
#             )
#             out[i] = O.transpose(0, 1)
#             lse[i] = LSE
#         return out, lse

#     out_flash, lse_flash = flash_kvcache()
#     out_torch, lse_torch = ref_kvcache()

#     print("Max abs diff out:", (out_flash - out_torch).abs().max().item())
#     cal_diff(out_flash, out_torch, "out")