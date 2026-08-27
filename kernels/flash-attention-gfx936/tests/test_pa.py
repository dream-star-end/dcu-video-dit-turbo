import argparse
import math
import random
from einops import rearrange, repeat
import torch
import triton

# from flash_attn import flash_attn_with_kvcache
from flash_attn import vllm_flash_attn_with_kvcache as flash_attn_with_kvcache
torch.set_printoptions(precision=4, profile="default", sci_mode=False)

def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    # print(query)
    # print(key)
    # print(attn_weight)
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    # print(attn_weight)
    # print(value)
    return attn_weight @ value


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (seqlen_k)
    sq = (seqlen_q)
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k)
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@torch.inference_mode()
def test_flash_kvcache(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen):
    # print(
    #     f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}"
    # )
    block_size = 64
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    # max_seqlen_pad=8192
    max_seqlen_pad=51200
    # if varlen:
    #     for i in range(b):
    #         # cache_seqlens[i] = mean_sk
    #         cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    #         # cache_seqlens[i] = (cache_seqlens[i] // 4) * 4
    #         # cache_seqlens[i] = 65
    # total_seqlens = cache_seqlens.sum().item()
    # mean_seqlens = cache_seqlens.float().mean().int().item()
    # max_seqlen = cache_seqlens.max().item()
    # max_seqlen_pad = triton.cdiv(max_seqlen, block_size) * block_size
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}, {max_seqlen_pad=}")

    q = torch.randn(b, s_q, h_q, d)
    
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, dv)

    # for i in range(64):
    #     for j in range(128):
    #         blocked_v[:, i, :, j] = i + 1
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
            float("nan")
        )
        blocked_v.view(b, max_seqlen_pad, h_kv, dv)[i, cache_seqlens[i].item():] = (
            float("nan")
        )

    blocked_k_ = blocked_k.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v.permute(0, 2, 3, 1).contiguous()
    # print(blocked_k_)
    # print(blocked_v_)
    def flash_kvcache():
        return flash_attn_with_kvcache(
            q,
            blocked_k_,
            blocked_v_,
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal=causal,
            num_splits = 0,
            max_seqlen_k=mean_sk,
            return_softmax_lse=False
        )

    def ref_kvcache():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
        return out.to(q.dtype)
    out_flash= flash_kvcache()
    out_torch= ref_kvcache()
    # print(out_flash)
    # print(out_torch)
    # print(out_flash-out_torch)
    print(torch.allclose(out_flash,out_torch,atol=1e-2, rtol=1e-2))


def main(torch_dtype, is_prof=False):
    device = torch.device("cuda:0")
    torch.set_printoptions(threshold=100000)
    torch.set_printoptions(precision=4)
    torch.set_printoptions(sci_mode=False)
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)
    # dim = [64,128,256]
    dim = [64]
    causal = True
    varlen = False
    for d in dim:
        test_flash_kvcache(1, 1, 1025, 16, 16, d, d, causal, varlen)
        test_flash_kvcache(20, 1, 16384, 28, 4, d, d, causal, varlen)
        test_flash_kvcache(8, 1, 17408, 16, 2, d, d, causal, varlen)
        test_flash_kvcache(240, 1, 2048, 6, 1, d, d, causal, varlen)
        test_flash_kvcache(32, 1, 3584, 32, 8, d, d, causal, varlen)
        test_flash_kvcache(64, 1, 3584, 16, 4, d, d, causal, varlen)
        test_flash_kvcache(70, 1, 3584, 4, 1, d, d, causal, varlen)
        test_flash_kvcache(80, 1, 3584, 32, 4, d, d, causal, varlen)
        test_flash_kvcache(100, 1, 3584, 16, 2, d, d, causal, varlen)
        test_flash_kvcache(80, 1, 3584, 8, 1, d, d, causal, varlen)
        test_flash_kvcache(24, 1, 2048, 8, 1, d, d, causal, varlen)
        test_flash_kvcache(32, 1, 2048, 8, 1, d, d, causal, varlen)
        test_flash_kvcache(48, 1, 2048, 8, 1, d, d, causal, varlen)
        test_flash_kvcache(48, 1, 1514, 10, 2, d, d, causal, varlen)
        test_flash_kvcache(2,  1, 32402,10, 2, d, d, causal, varlen)
        test_flash_kvcache(1,  1, 51200,10, 2, d, d, causal, varlen)
        test_flash_kvcache(1, 1, 1235, 17, 1, d, d, causal, varlen)
        test_flash_kvcache(1, 1, 455, 24, 1, d, d, causal, varlen)
        test_flash_kvcache(1, 1, 256, 8, 4, d, d, causal, varlen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="fp16",
        help="Data type to use for testing (bf16 or fp16)",
    )
    parser.add_argument('--prof', default=False, action='store_true', help='prof or not')

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    # main(torch_dtype, args.prof)
    print("test bf16")
    main(torch.bfloat16, args.prof)
    print("test fp16")
    main(torch.float16, args.prof)