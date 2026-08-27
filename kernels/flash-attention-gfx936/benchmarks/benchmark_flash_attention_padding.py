      
# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# from flash_attn import flash_attn_func_blasst as flash_attn_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--prof', default=False, action='store_true', help='prof or not')
parser.add_argument('--bhsd', default=False, action='store_true', help='bhsd or not')
parser.add_argument('--hy', default=False, action='store_true', help='hy code or not')
parser.add_argument('--ali', default=False, action='store_true', help='alibaba size or not')
parser.add_argument('--qwen', default=False, action='store_true', help='qwen size or not')
parser.add_argument('--xf', default=False, action='store_true', help='xunfei size or not')
parser.add_argument('--fwd', default=False, action='store_true', help='only run fwd')
args = parser.parse_args()

# def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
#     assert mode in ["fwd", "bwd", "fwd_bwd"]
#     f = 4 * batch * seqlen ** 2 * nheads * headdim
#     if causal:
#         f = f / 2
#     return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def flops(batch, seqlen, headdim, headdimv, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    # f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    f = 2 * batch * seqlen**2 * nheads * (headdim + headdimv) // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10 ** 12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


def time_forward(func, *args, **kwargs):
    _, time_b = benchmark_forward(func, *args, **kwargs)
    return time_b.mean


def padding_bmhk(t):  # BMHK
    # print(f"padding..")
    batch, seqlen, nheads, dim = t.shape
    t_tmp = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads * dim), (0, 32), 'constant', 0)[:, :, :-32].reshape(batch, seqlen, nheads, dim)
    # print(f"{t_tmp.shape=}, {t_tmp.stride()=}")
    return t_tmp


repeats = 30
device = 'cuda'
dtype = torch.bfloat16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [(128, 128)]
# headdim_vals = [160, 192, 224, 256]
nheads_vals = [(16, 16)]
window_size = (-1, -1)

if args.qwen:
    bs_seqlen_vals = [(2, 256), (2, 384), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), 
                      (2, 1920), (2, 2048), (2, 2304), (2, 2432), (2, 2944), (2, 3456), (2, 3584), (2, 3712), (2, 3968), (2, 4096)]
    causal_vals = [causal_vals[-1]]
    nheads_vals = [(32, 32)]

if args.ali:
    bs_seqlen_vals = [(1, 8192)]
    causal_vals = [causal_vals[-1]]
    nheads_vals = [(16, 16), (32, 32), (32, 4), (52, 4), (16, 2), (26, 2), (8, 1), (13, 1)]

if args.xf:
    bs_seqlen_vals = bs_seqlen_vals # [(2, 8192)]
    causal_vals = [causal_vals[-1]]
    nheads_vals = [(8, 2)]
    window_size = (8191, 0)

if args.prof:
    repeats = 1
    bs_seqlen_vals = [bs_seqlen_vals[-1]]
    causal_vals = [causal_vals[-2]]

bhsd = False
if args.bhsd or args.hy:
    bhsd = True

dropout_p = 0.0
pad = 0
methods = (["Flash2"])

fwdOnly = args.fwd
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for nheads_q, nheads_k in nheads_vals:
    for causal in causal_vals:
        for headdim, headdimv in headdim_vals:
            for batch_size, seqlen in bs_seqlen_vals:
                config = (causal, headdim, headdimv, batch_size, seqlen)

                if not bhsd:
                    q = torch.randn(batch_size, seqlen, nheads_q, headdim + pad, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch_size, seqlen, nheads_k, headdim + pad, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch_size, seqlen, nheads_k, headdimv + pad, device=device, dtype=dtype, requires_grad=True)
                    # q = q[:, :, :, :headdim]
                    # k = k[:, :, :, :headdim]
                    # v = v[:, :, :, :headdim]
                    # q = q.as_strided(q.size(), [seqlen * nheads_q * headdim, headdim, headdim * nheads_q, 1])
                    # k = k.as_strided(k.size(), [seqlen * nheads_k * headdim, headdim, headdim * nheads_k, 1])
                    # v = v.as_strided(k.size(), [seqlen * nheads_k * headdim, headdim, headdim * nheads_k, 1])
                else:
                    q = torch.randn(batch_size, nheads_q, seqlen, headdim + pad, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch_size, nheads_k, seqlen, headdim + pad, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch_size, nheads_k, seqlen, headdimv + pad, device=device, dtype=dtype, requires_grad=True)

                if fwdOnly:
                    if args.hy:
                        f = time_forward(flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False)
                    else:
                        f = time_forward(flash_attn_func, q, k, v, dropout_p, causal=causal, bhsd=bhsd, window_size=window_size, repeats=repeats, verbose=False)
                    time_f[config, "Flash2"] = f
                else:
                    if args.hy:
                        f, b = time_fwd_bwd(flash_attn_func, q, k, v, dropout_p, causal=causal, repeats=repeats, verbose=False)
                    else:
                        f, b = time_fwd_bwd(flash_attn_func, q, k, v, dropout_p, causal=causal, bhsd=bhsd, window_size=window_size, repeats=repeats, verbose=False)
                    time_f[config, "Flash2"] = f
                    time_b[config, "Flash2"] = b

                print(f"### causal={causal}, headdim={headdim}, headdim={headdimv}, batch_size={batch_size},nheads={nheads_q}, seqlen={seqlen} ###")
                nheads = nheads_q

                for method in methods:
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, headdimv, nheads_q, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    if fwdOnly:
                        print(
                            f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1000:.2f} ms. "
                        )
                    else:
                        time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                        speed_b[config, method] = efficiency(
                            flops(batch_size, seqlen, headdim, headdimv, nheads, causal, mode="bwd"),
                            time_b[config, method]
                        )
                        speed_f_b[config, method] = efficiency(
                            flops(batch_size, seqlen, headdim, headdimv, nheads, causal, mode="fwd_bwd"),
                            time_f_b[config, method]
                        )
                        print(
                            f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, {time_f[config, method] * 1000:.2f} ms. "
                            f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, {time_b[config, method] * 1000:.2f} ms. "
                            f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s, {time_f_b[config, method] * 1000:.2f} ms. "
                        )

    
