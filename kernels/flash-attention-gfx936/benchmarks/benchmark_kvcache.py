import pickle
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from openpyxl import Workbook
from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func,flash_attn_func
from flash_attn import flash_attn_varlen_func

# wb = Workbook()
# ws = wb.active


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--prof', default=False, action='store_true', help='prof or not')
parser.add_argument('--fwd', default=False, action='store_true', help='only run fwd')
args = parser.parse_args()


def flops(batch, seqlen, nheads, seqlen_k, nheads_kv, headdim, headdimv, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    # f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    f = 2 * batch * seqlen * seqlen_k * nheads * (headdim + headdimv) // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_forward(func, *args, **kwargs):
    time_f, time_b = benchmark_forward(func, *args, **kwargs)
    return time_b.mean

def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def padding_bmhk(t):   # BMHK 
    # print(f"padding..")
    batch, seqlen, nheads, dim = t.shape
    t_tmp = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads*dim), (0, 32), 'constant', 0)[:,:,:-32].reshape(batch, seqlen, nheads, dim)
    # print(f"{t_tmp.shape=}, {t_tmp.stride()=}")
    return t_tmp

repeats = 30
device = 'cuda'
dtype = torch.float16
dropout_p =0.0
pad=0
methods = (["Flash2"])
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}


fwdOnly = args.fwd

# ws.append(['batch_size', 'total_q', 'total_kv', 'nheads_q', 'num_heads_kv', 'causal', 'dim', 'dimv', 'dtype', 'tflops', 'time(ms)'])
test_size = [
    (32, 512, 32, 512, 8, 128, 128, True),
    (16, 1024, 32, 1024, 8, 128, 128, True),
    (8, 2048, 32, 2048, 8, 128, 128, True),
    (4, 4096, 32, 4096, 8, 128, 128, True),
    (2, 8192, 32, 8192, 8, 128, 128, True),
    (1, 16384, 32, 16384, 8, 128, 128, True),
]

if args.prof:
    repeats = 1
    test_size = [test_size[-1]]
for batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal in test_size:
    config = (batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal)
    q = torch.randn(batch_size, total_q, nheads_q , headdim, device=device, dtype=dtype, requires_grad=False)
    # k = torch.randn(batch_size, total_kv, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    # v = torch.randn(batch_size, total_kv, nheads_k, headdimv, device=device, dtype=dtype, requires_grad=True)
    # q = padding_bmhk(q)
    # k = padding_bmhk(k)
    # v = padding_bmhk(v)
    # # print(q.shape)
    # print(q.stride())
    block_size = 64
    q = q.reshape(batch_size*total_q, nheads_q, headdim)
    # 初始化KV Cache和块表
    num_blocks = math.ceil(total_kv / block_size) * batch_size 
    # num_blocks = (total_kv + block_size - 1) // block_size
    k_cache = torch.randn(num_blocks, block_size, nheads_k, headdim, device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, nheads_k, headdimv, device=device, dtype=dtype)

    # k_cache = padding_bmhk(k_cache)
    # v_cache = padding_bmhk(v_cache)

    # block_table = torch.zeros(batch_size, num_blocks, dtype=torch.int32, device=device)
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    # k = k.reshape(batch_size*total_kv, nheads_k, headdim)
    # v = v.reshape(batch_size*total_kv, nheads_k, headdimv)
            # q=query,
            # k=key_cache,
            # v=value_cache,
            # cu_seqlens_q=cu_query_lens,
            # cu_seqlens_k=cu_kv_lens,
            # max_seqlen_q=max_query_len,
            # max_seqlen_k=max_kv_len,
            # softmax_scale=scale,
            # causal=True,
            # window_size=window_size,
            # block_table=block_tables,
            # softcap=soft_cap if soft_cap is not None else 0,

    cu_seqlens = torch.arange(0, (batch_size + 1) * total_kv, step=total_q, dtype=torch.int32,
                        device=device)
    # if fwdOnly:
    f = time_forward(
        flash_attn_varlen_func, q, k_cache, v_cache, cu_seqlens, cu_seqlens, total_q, total_kv, dropout_p,
        block_table=block_table,
        causal=causal, repeats=repeats, verbose=False
    )
    time_f[config, "Flash2"] = f
    # else:
    #     f, b = time_fwd_bwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, total_q, total_kv, dropout_p,
    #         causal=causal, repeats=repeats, verbose=False)
    #     time_f[config, "Flash2"] = f
    #     time_b[config, "Flash2"] = b
    print(f"### causal={causal}, headdim={headdim}, headdimv={headdimv}, batch_size={batch_size}, nheads_q={nheads_q}, nheads_k={nheads_k}, total_q={total_q}, total_kv={total_kv} ###")
    for method in methods:
        # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
        speed_f[config, method] = efficiency(
            flops(batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal, mode="fwd"),
            time_f[config, method]
        )
        print(
            f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1000:.2f} ms. "
        )
        


        # ws.append([batch_size, seqlen, seqlen, nheads_q, nheads_k, causal, headdim, "float16", round(speed_f[config, method], 2), round(time_f[config, method]*1000, 2)])
        # exit(0)

# wb.save("varlen_64_32_4_018a7dd_waq.xlsx")

