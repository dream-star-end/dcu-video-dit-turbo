import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from openpyxl import Workbook
from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func,flash_attn_func
from flash_attn import flash_attn_varlen_func

wb = Workbook()
ws = wb.active



def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim 
    if causal:
        f=f/2
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def time_forward(func, *args, **kwargs):
    time_f, time_b = benchmark_forward(func, *args, **kwargs)
    return time_b.mean

def padding_bmhk(t):   # BMHK 
    # print(f"padding..")
    batch, seqlen, nheads, dim = t.shape
    t_tmp = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads*dim), (0, 32), 'constant', 0)[:,:,:-32].reshape(batch, seqlen, nheads, dim)
    # print(f"{t_tmp.shape=}, {t_tmp.stride()=}")
    return t_tmp

repeats = 30
device = 'cuda'
dtype = torch.float16
bs_seqlen_vals = [(1,128), (1, 1024), (1, 2048), (1, 4096), (1, 6144), (1, 8192), (1, 10*1024), (1, 12*1024), (1, 16*1024), (1, 32*1024), (1, 64*1024)]
# bs_seqlen_vals = [(1, 1024), (1, 2048), (1, 4096), (1, 8192), (1, 16*1024), (1, 32*1024)]
# bs_seqlen_vals += [(8, 1024), (8, 2048), (8, 4096), (8, 8192), (8, 16*1024), (8, 32*1024)]
# bs_seqlen_vals += [(16, 2049), (32, 1024), (64, 512), (128, 256), (256, 128)]
causal_vals = [True]
headdim_vals = [128]
nheads_vals = [(32, 2), (16, 1), (8, 1), (32, 8),
    (32, 32), (16, 16), (8, 8), (4, 4), (40, 40),
    (20, 20), (10, 10), (5, 5), (32, 4), (16, 2), (16, 16), 
    (14, 2), (7, 1), (20, 4), (10, 2), (5, 1)]
# nheads_vals=[(28,4)]
dropout_p =0.0
pad=0
methods = (["Flash2"])
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

# ws.append(['batch_size', 'total_q', 'total_kv', 'nheads_q', 'num_heads_kv', 'causal', 'dim', 'dtype', 'tflops', 'time(ms)'])

for batch_size, seqlen in bs_seqlen_vals:
    for causal in causal_vals:
        for headdim in headdim_vals:
            for nheads_q, nheads_k in nheads_vals:
                config = (causal, headdim, batch_size, seqlen, nheads_q, nheads_k)
                q = torch.randn(batch_size,seqlen, nheads_q , headdim, device=device, dtype=dtype, requires_grad=True)
                k = torch.randn(batch_size,seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
                v = torch.randn(batch_size,seqlen, nheads_k, headdim, device=device, dtype=dtype, requires_grad=True)
                q = padding_bmhk(q)
                k = padding_bmhk(k)
                v = padding_bmhk(v)
                # # print(q.shape)
                # print(q.stride())
                q = q.reshape(batch_size*seqlen, nheads_q, headdim)
                k = k.reshape(batch_size*seqlen, nheads_k, headdim)
                v = v.reshape(batch_size*seqlen, nheads_k, headdim)

                # print(q.shape)
                # print(q.stride())
                # print(k.shape)
                # print(k.stride())
                # print(v.shape)
                # exit(-1)
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                    device=device)

                f = time_forward(
                    flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p,
                    causal=causal, repeats=repeats, verbose=False
                )

                time_f[config, "Flash2"] = f
                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, nheads_q={nheads_q}, nheads_k={nheads_k}, seqlen={seqlen} ###")
                for method in methods:
                    # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads_q, causal, mode="fwd"),
                        time_f[config, method]
                    )

                    print(
                        f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s,  {time_f[config, method]*1000:.2f} ms"
                        # f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                        # f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                    )

                    # ws.append([batch_size, seqlen, seqlen, nheads_q, nheads_k, causal, headdim, "float16", round(speed_f[config, method], 2), round(time_f[config, method]*1000, 2)])
                    # exit(0)

# wb.save("varlen_64_32_4_018a7dd_waq.xlsx")
