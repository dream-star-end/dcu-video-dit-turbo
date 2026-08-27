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
    (32, 512, 16, 512, 16, 192, 128, False),
    (16, 1024, 16, 1024, 16, 192, 128, False),
    (8, 2048, 16, 2048, 16, 192, 128, False),
    (4, 4096, 16, 4096, 16, 192, 128, False),
    (2, 8192, 16, 8192, 16, 192, 128, False),
    (1, 16384, 16, 16384, 16, 192, 128, False),
    (32, 512, 16, 512, 16, 192, 128, True),
    (16, 1024, 16, 1024, 16, 192, 128, True),
    (8, 2048, 16, 2048, 16, 192, 128, True),
    (4, 4096, 16, 4096, 16, 192, 128, True),
    (2, 8192, 16, 8192, 16, 192, 128, True),
    (1, 16384, 16, 16384, 16, 192, 128, True),
]

if args.prof:
    repeats = 1
    test_size = [test_size[-3]]
for batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal in test_size:
    config = (batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal)
    q = torch.randn(batch_size, total_q, nheads_q , headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, total_kv, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, total_kv, nheads_k, headdimv, device=device, dtype=dtype, requires_grad=True)
    # q = padding_bmhk(q)
    # k = padding_bmhk(k)
    # v = padding_bmhk(v)
    # # print(q.shape)
    # print(q.stride())
    q = q.reshape(batch_size*total_q, nheads_q, headdim)
    k = k.reshape(batch_size*total_kv, nheads_k, headdim)
    v = v.reshape(batch_size*total_kv, nheads_k, headdimv)

    # print(q.shape)
    # print(q.stride())
    # print(k.shape)
    # print(k.stride())
    # print(v.shape)
    # exit(-1)
    cu_seqlens = torch.arange(0, (batch_size + 1) * total_q, step=total_q, dtype=torch.int32,
                        device=device)
    if fwdOnly:
        f = time_forward(
            flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, total_q, total_kv, dropout_p,
            causal=causal, repeats=repeats, verbose=False
        )
        time_f[config, "Flash2"] = f
    else:
        f, b = time_fwd_bwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, total_q, total_kv, dropout_p,
            causal=causal, repeats=repeats, verbose=False)
        time_f[config, "Flash2"] = f
        time_b[config, "Flash2"] = b
    print(f"### causal={causal}, headdim={headdim}, headdimv={headdimv}, batch_size={batch_size}, nheads_q={nheads_q}, nheads_k={nheads_k}, total_q={total_q}, total_kv={total_kv} ###")
    for method in methods:
        # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
        speed_f[config, method] = efficiency(
            flops(batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal, mode="fwd"),
            time_f[config, method]
        )
        if fwdOnly:
            print(
                f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1000:.2f} ms. "
            )
        else:
            time_f_b[config, method] = time_f[config, method] + time_b[config, method]
            speed_b[config, method] = efficiency(
                flops(batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal, mode="bwd"),
                time_b[config, method]
            )
            speed_f_b[config, method] = efficiency(
                flops(batch_size, total_q, nheads_q, total_kv, nheads_k, headdim, headdimv, causal, mode="fwd_bwd"),
                time_f_b[config, method]
            )
            print(
                f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1000:.2f} ms. "
                f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, {time_b[config, method] * 1000:.2f} ms. "
                f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s, {time_f_b[config, method] * 1000:.2f} ms. "
            )

        # ws.append([batch_size, seqlen, seqlen, nheads_q, nheads_k, causal, headdim, "float16", round(speed_f[config, method], 2), round(time_f[config, method]*1000, 2)])
        # exit(0)

# wb.save("varlen_64_32_4_018a7dd_waq.xlsx")
