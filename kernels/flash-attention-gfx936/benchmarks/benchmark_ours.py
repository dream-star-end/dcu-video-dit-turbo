import os
import math
import numpy
import torch
import torch.utils.benchmark as benchmark
from collections import namedtuple
import argparse


def flops(batch, seq_len, headdim, qheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if (isinstance(seq_len, int)):
        f = 4 * batch * seq_len**2 * qheads * headdim // (2 if causal else 1)
    else:
        tmp = sum([(seq_len[k + 1] - seq_len[k])**2 for k in range(len(seq_len) - 1)])
        f = 4 * tmp * qheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def benchmark_forward(
    fn, *inputs, repeats=100, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def efficiency(flop, time):
    return (flop / time / 10**12)

def warp_tensor(tensor, gpu_is_ours, is_varlen=False, num_head=None):
    if (not is_varlen):
        return tensor if (gpu_is_ours) else tensor.transpose(1, 2).contiguous()
    else:
        return tensor if (gpu_is_ours) else tensor.view(-1, num_head, tensor.shape[-1])


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--repeats',    default=1, type=int, help='run times during once benchmark')
parser.add_argument('--iterations', default=6, type=int, help='times of benchmark')
parser.add_argument('--compare', default=None, type=str, help='competitor card name')
parser.add_argument('--ratio', default=False, action='store_true', help='whether compute ratio of ours/nvidia')
args = parser.parse_args()

# prepare testing cases
params = namedtuple('param', ['causal', 'batch_size','qheads','kvheads','seq_len','head_size','window_size'])
params_list = [
    params(batch_size=4, qheads=32, kvheads=32, seq_len=(0, 1000, 2000, 3000, 4000), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=2, qheads=32, kvheads=32, seq_len=(0, 2000, 4000), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=4, qheads=16, kvheads=2, seq_len=(0, 1000, 2000, 3000, 4000), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=2, qheads=16, kvheads=2, seq_len=(0, 2000, 4000), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=16, kvheads=2, seq_len=(0, 20000), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=16, kvheads=2, seq_len=(0, 20305), head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=16, kvheads=16, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=32, kvheads=32, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=32, kvheads=4, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=52, kvheads=4, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=16, kvheads=2, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=26, kvheads=2, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=8, kvheads=1, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
    params(batch_size=1, qheads=13, kvheads=1, seq_len=8192, head_size=128, causal=True, window_size=[-1,-1]),
]

import flash_attn
import flash_attn_2_cuda as _C_flashattention
print("load flash_attn from package")
gpu_card_info = torch.cuda.get_device_properties(0)
gpu_is_ours   = bool("NVIDIA" not in gpu_card_info.name)


speed_on_this_gpu = []
for idx, params in enumerate(params_list):
    torch.cuda.empty_cache()
    cost_time   = []
    device      = "cuda"
    causal      = params.causal
    batch_size  = params.batch_size
    qheads      = params.qheads
    kvheads     = params.kvheads
    seq_len     = params.seq_len
    head_size   = params.head_size
    window_size = params.window_size
    flops_count = flops(batch_size, seq_len, head_size, qheads, causal)
    repeats     = args.repeats
    iterations  = args.iterations
    is_varlen   = isinstance(seq_len, tuple)
    for i in range(iterations):
        torch.cuda.empty_cache()
        if (is_varlen):
            max_seqlen_q   = max([seq_len[k + 1] - seq_len[k] for k in range(len(seq_len) - 1)])
            seq_len        = torch.tensor(list(seq_len), dtype=torch.int32).cuda()
            total_seqlen_q = seq_len[-1].item()
            q = warp_tensor(torch.randn(qheads * total_seqlen_q, head_size, device=device,dtype=torch.float16), gpu_is_ours, is_varlen, qheads)
            k = warp_tensor(torch.randn(kvheads * total_seqlen_q, head_size, device=device,dtype=torch.float16), gpu_is_ours, is_varlen, kvheads)
            v = warp_tensor(torch.randn(kvheads * total_seqlen_q, head_size, device=device,dtype=torch.float16), gpu_is_ours, is_varlen, kvheads)
            if ("2.6" in str(flash_attn.__version__)):
                fa_varlen_args = (q, k, v, None, seq_len, seq_len, None, None, max_seqlen_q, max_seqlen_q, 0.0, 1.0 / math.sqrt(head_size), False, causal, window_size[0], window_size[1], 0.0, False, None)
            else:
                fa_varlen_args = (q, k, v, None, seq_len, seq_len, None, None, max_seqlen_q, max_seqlen_q, 0.0, 1.0 / math.sqrt(head_size), False, causal, window_size[0], window_size[1], False, None)
            t = benchmark_forward(_C_flashattention.varlen_fwd, *fa_varlen_args, repeats=repeats, verbose=False)[1].times[0]
        else:
            q = warp_tensor(torch.randn(batch_size, qheads,  seq_len, head_size, device=device,dtype=torch.float16, requires_grad=True), gpu_is_ours)
            k = warp_tensor(torch.randn(batch_size, kvheads, seq_len, head_size, device=device,dtype=torch.float16, requires_grad=True), gpu_is_ours)
            v = warp_tensor(torch.randn(batch_size, kvheads, seq_len, head_size, device=device,dtype=torch.float16, requires_grad=True), gpu_is_ours)
            t = benchmark_forward(flash_attn.flash_attn_interface.flash_attn_func, q, k, v, 0.0, causal=causal, window_size=window_size, repeats=repeats, verbose=False)[1].times[0]
        if(i > 0):
            cost_time.append(t)
            # print("{:.9f}  {:.9f}".format(t, efficiency(flops_count, t)))
        # delete the data each time to avoid detecting the cache
        del q, k, v
    cost_time = numpy.array(cost_time)
    cost_time_mean = cost_time.mean()
    # remove bursts of dirty data
    cost_time = numpy.delete(cost_time, numpy.where(cost_time < (0.8 * cost_time_mean)))
    cost_time_mean = cost_time.mean()
    speed = efficiency(flops_count, cost_time_mean)
    speed_on_this_gpu.append(speed)

if (gpu_is_ours):
    if (args.ratio):
        for it in speed_on_this_gpu:
            print(it)
        exit()
    # prepare performance sheet for comparison
    nvidia_performance = {
        # for L20, the numerical value of "repeat" has very little effect, and thus only one piece of data. "repeats" of 100 is adopted
        "L20": [81.95, 89.90, 74.01, 81.75, 108.61, 108.59, 101.95, 106.80, 106.89, 108.62, 102.55, 105.85, 94.71, 100.60],
        # for A800, the numerical value of "repeat" has very significant effect, and thus several pieces of data.
        "A800": [103.01, 130.44, 78.70, 99.94, 203.21, 203.51, 191.49, 204.63, 207.69, 213.23, 192.70, 204.25, 163.50, 185.51],
    }
    # acquire corresponding card
    if (args.compare is not None):
        nvidia_competitor = args.compare
        if (nvidia_competitor not in nvidia_performance.keys()):
            print("\033[1;31mPerformance of competitor is not recorded yet!\033[0m".format(nvidia_competitor))
        nvidia_speed = nvidia_performance[nvidia_competitor]
    else:
        nvidia_competitor = "A800"
        nvidia_speed = nvidia_performance[nvidia_competitor]
    # check data alignment
    if (len(nvidia_speed) != len(speed_on_this_gpu)):
        print("\x1b[31mPerformance data of ours and {} is not correct\x1b[0m\n\n".format(nvidia_competitor))
        exit()
    # output info
    speed_ratio = []
    print("ours             {}             Ratio".format(nvidia_competitor))
    for i, (ours, nvidia) in enumerate(zip(speed_on_this_gpu, nvidia_speed)):
        print("{:.9f}\t{:.9f}\t{:.2f}%".format(ours, nvidia, ours / nvidia * 100))
        speed_ratio.append(ours / nvidia)
    speed_on_this_gpu    = numpy.array(speed_on_this_gpu)
    nvidia_speed = numpy.array(nvidia_speed)
    speed_ratio  = numpy.array(speed_ratio)
    print("============================================")
    print("{:.9f}\t{:.9f}\t{:.2f}%".format(speed_on_this_gpu.mean(), nvidia_speed.mean(), speed_ratio.mean() * 100))
    print("Mean  of ours      : {:.9f}".format(speed_on_this_gpu.mean()))
    print("Mean  of NVIDIA {}: {:.9f}".format(nvidia_competitor, nvidia_speed.mean()))
    print("Ratio to NVIDIA {}: \x1b[32m{:.2f}%\x1b[0m\n\n".format(nvidia_competitor, 100 * speed_ratio.mean()))

else:
    for it in speed_on_this_gpu:
        print(it)