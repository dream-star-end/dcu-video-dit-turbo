import argparse
import math
import random

import torch
import triton
import pdb
# import flash_attn_2_cuda as flash_attn_cuda

from flash_attn import vllm_flash_attn_with_kvcache
torch.set_printoptions(precision=4, profile="default", sci_mode=False)

def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    tmp = query @ key.transpose(-2, -1)
    # print("attn_weight ", tmp[0, 0, :10])
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
    # print("  ", key[0])
    # print("k_scale ", k_scale[0, :8])
    # print(" key k_scale ", key.shape, k_scale.shape)
    # print(" key ", key.shape)
    # key = key * k_scale
    # print("key ", key[0, 0:2, :8])
    # value = value * v_scale
    # print("k_scale ", k_scale[0:2, :8])
    for i in range(key.shape[0]):
        key[i] = key[i] * k_scale[i]
        value[i] = value[i] * v_scale[i]
    # print("key ", key[0:2, 0, :8])
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    # k_scale = k_scale.repeat_interleave(h_q // h_kv, dim=0)
    # v_scale = v_scale.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight_temp = query @ key.transpose(-2, -1) 
    # print(" attn_weight_temp  ", attn_weight_temp[0, :3, :4])
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


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    torch_dtype = x.dtype
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < (1e-4 if torch_dtype == torch.bfloat16 else 1e-5)


@torch.inference_mode()
def test_flash_kvcache(b, s_q, mean_sk, h_q, h_kv, d, causal, varlen, is_prof=False):
    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {causal=}, {varlen=}"
    )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    # cache_seqlens[0] = 127
    # print(" cache_seqlens[i] ",  cache_seqlens)
    
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 64) * 64
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}, {max_seqlen_pad=}")

    q = torch.randn(b, s_q, h_q, d)
    # q[0, 0, 0, 0] = 2
    # q[:, :, :, 0:32] = 0
    # q[:, :, :, 32:64] = 0
    # q[:, :, :, 64:96] = 0
    # q[:, :, :, 96:128] = 0
    # for j in range(d):
    #     q[0, :, 0, j] = j 
    block_size = 64
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    # blocked_k[0, 0, 0, 0] = 1
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, d)
    # pad = 0
    # blocked_k = torch.nn.functional.pad(
    #     blocked_k.reshape(
    #         block_table.numel(), block_size, h_kv*d), 
    #         (0, pad), 'constant', 0)[:,:,:-pad].reshape(block_table.numel(), block_size, h_kv, d)
    # for i in range(b):
    #     blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
    #         # float("nan")
    #         0
    #     )
    #     blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
    #         # float("nan")
    #         0
    #     )
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )
        blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )


    blocked_k_ = blocked_k.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v.permute(0, 2, 3, 1).contiguous()
    def flash_kvcache():
        return vllm_flash_attn_with_kvcache(
            q = q,
            k_cache = blocked_k_,
            v_cache = blocked_v_,
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal = causal,
            return_softmax_lse = True,
            num_splits = 0,
        )

    def ref_kvcache():
        out = torch.empty(b, s_q, h_q, d, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
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
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    # # out_flash = flash_kvcache()
    out_flash, lse_flash = flash_kvcache()
    # if is_prof: return
    out_torch, lse_torch = ref_kvcache()

    # print("lse_flash:", lse_flash[0, 0, :16])
    # print("lse_torch:", lse_torch[0, 0, :16])
    # print("out_flash:", out_flash[0, 0, 0, :16])
    # print("out_torch:", out_torch[0, 0, 0, :16])
    # indexs = torch.nonzero((out_flash - out_torch).abs() > 0.01)
    # # print("indexs ", indexs)
    # print("nan ", torch.nonzero(torch.isnan(out_flash)))
    # # pdb.set_trace()
    print("lse_flash - lse_torch", (lse_torch - lse_flash).abs().max())
    print("out_torch - out_flash", (out_flash - out_torch).abs().max())
    cal_diff(lse_flash, lse_torch, "lse")
    cal_diff(out_flash, out_torch, "out")
    # cal_diff(lse_flash, lse_torch, "lse")

    t = triton.testing.do_bench(flash_kvcache)
    print(
        f"{t:.3f} ms"
    )

@torch.inference_mode()
def test_flash_kvcache_int8(b, s_q, mean_sk, h_q, h_kv, d, causal, varlen, is_prof=False):
    print(
        f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {causal=}, {varlen=}"
    )

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 64) * 64
    print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}, {max_seqlen_pad=}")

    q = torch.randn(b, s_q, h_q, d)
    # q = torch.ones(b, s_q, h_q, d)
    # for i in range(s_q):
    #     for j in range(d):
    #         q[0, i, 0, j] = i
    # q[0, 0, 0, 0] = 1
    block_size = 64
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randint(low=-10, high=10, size = (block_table.numel(), block_size, h_kv, d), dtype = torch.int8).to(torch.int8)
    blocked_v = torch.randint(low=-10, high=10, size = (block_table.numel(), block_size, h_kv, d), dtype = torch.int8).to(torch.int8)
    # blocked_k = torch.ones(size = (block_table.numel(), block_size, h_kv, d), dtype = torch.int8).to(torch.int8) * 1
    # blocked_v = torch.ones(size = (block_table.numel(), block_size, h_kv, d), dtype = torch.int8).to(torch.int8)
    # blocked_k[0, 0, 0, 0] = 1
    # blocked_k[0, 1, 0, 0] = 2
    # blocked_k[0, 2, 0, 0] = 3
    # blocked_k[0, 3, 0, 0] = 4
    # blocked_k[0, 4, 0, 0] = 5
    # print(blocked_k[0, 0, 0, :3])
    # for i in range(64):
    #     for j in range(128):
    #         blocked_k[:, i, :, j] = i
    k_scale = torch.randn(h_kv, d, dtype = torch.float)
    v_scale = torch.randn(h_kv, d, dtype = torch.float)
    
    # k_scale = torch.ones(h_kv, d, dtype = torch.float)
    # v_scale = torch.ones(h_kv, d, dtype = torch.float)
    # for i in range(128):
    #     v_scale[:, i] = i
    # k_scale[0]
    # for i in range(128):
    #     k_scale[:, i] = i
    # print("k_scale ", k_scale)
    # pad = 0
    # blocked_k = torch.nn.functional.pad(
    #     blocked_k.reshape(
    #         block_table.numel(), block_size, h_kv*d), 
    #         (0, pad), 'constant', 0)[:,:,:-pad].reshape(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            -128
        )
        blocked_v.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            -128
        )


    blocked_k_ = blocked_k.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v.permute(0, 2, 3, 1).contiguous()

    # print("blocked_k_ ", blocked_k_[0, 0, :4, 0])
    def flash_kvcache():
        return vllm_flash_attn_with_kvcache(
            q = q,
            k_cache = blocked_k_,
            v_cache = blocked_v_,
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal = causal,
            return_softmax_lse = True,
            k_scale = k_scale,
            v_scale = v_scale,
            num_splits = 0,
            # softmax_scale = 0.3,
        )
    # print(" key k_scale ", blocked_k.view(-1, h_kv, d)[1:4].transpose(0, 1).shape, k_scale.shape)
    # def ref_kvcache():
    #     out = torch.empty(b, s_q, h_q, d, dtype=torch.float32)
    #     lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
    #     for i in range(b):
    #         begin = i * max_seqlen_pad
    #         end = begin + cache_seqlens[i]
    #         O, LSE = scaled_dot_product_attention_int8(
    #             q[i].transpose(0, 1),
    #             blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
    #             blocked_v.view(-1, h_kv, d)[begin:end].transpose(0, 1),
    #             h_q=h_q,
    #             h_kv=h_kv,
    #             k_scale = k_scale,
    #             v_scale = v_scale,
    #             is_causal=causal,
               
    #         )
    #         out[i] = O.transpose(0, 1)
    #         lse[i] = LSE
    #     return out, lse

    # out_flash, lse_flash = flash_kvcache()
    # if is_prof: return
    # out_torch, lse_torch = ref_kvcache()
    # print("out_torch ", out_torch[0, 0, 0, :10])
    # print("out_flash ", out_flash[0, 0, 0, :10])
    
    # print("lse_torch ", lse_torch[0, 0, :10])
    # print("lse_flash ", lse_flash[0, 0, :10])
    # # # print("out_flash:", out_flash)
    # # # print("out_torch:", out_torch)
    # # print("lse flash diff ", torch.nonzero((lse_flash - lse_torch).abs() > 0.01))
    # print(torch.nonzero((out_flash - out_torch).abs() > 1))
    # # pdb.set_trace()
    # print("out_flash diff", (out_flash - out_torch).max().item())
    # print("lse_flash diff", (lse_flash - lse_torch).max().item())
    # cal_diff(lse_flash, lse_torch, "lse")
    # cal_diff(out_flash, out_torch, "out")
    
    def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
        assert mode in ["fwd", "bwd", "fwd_bwd"]
        f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
        return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)
    def efficiency(flop, time):
        return (flop / time / 10**9) if not math.isnan(time) else 0.0

    t = triton.testing.do_bench(flash_kvcache)
    FLOPS = s_q * total_seqlens * h_q * (d + d) * 2
    # FLOPS = FLOPS // 2 if causal else FLOPS
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * d) * (
        torch.finfo(q.dtype).bits // 8
    )
    # print(
    #     f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s"
    # )
    print(f"{t:.3f} ms")


def main(torch_dtype, is_prof=False):
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    '''
    h_kv = 1
    d, dv = 576, 512
    causal = True

    for b in [128]:
        for s in [4096, 8192]:
            for h_q in [16, 32, 64, 128]:  # TP = 8, 4, 2, 1
                for s_q in [1, 2]:  # MTP = 1, 2
                    for varlen in [False, True]:
                        test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen)
    #                b, s_q,    s,   h_q, h_kv,   d,  dv, causal, varlen'''
    test_flash_kvcache(   32,   512,   512,    32,    8, 128,   True,   True, is_prof=is_prof)
    test_flash_kvcache(   16,   1024,   1024,    32,    8, 128,   True,   True, is_prof=is_prof)
    test_flash_kvcache(   8,    2048,   2048,    32,    8, 128,   True,   True, is_prof=is_prof)
    test_flash_kvcache(   4,    4096,   4096,    32,    8, 128,   True,   True, is_prof=is_prof)
    test_flash_kvcache(   2,    8192,   8192,    32,    8, 128,   True,   True, is_prof=is_prof)
    # test_flash_kvcache(   1,   16384,   16384,    16,    16, 128,   True,   True, is_prof=is_prof)

    '''
    h_kv = 1
    d, dv = 128, 128
    causal = True

    for b in [1, 32]:
        for s in [200, 1002, 2002, 1024, 2000, 4000, 32768, 65536]:
            for h_q in [4]:
                for s_q in [1]:  # MTP = 1, 2
                    for varlen in [True]:
                        test_flash_kvcache(b, s_q, s, h_q, h_kv, d, causal, varlen)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )
    parser.add_argument('--prof', default=False, action='store_true', help='prof or not')

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype, args.prof)
