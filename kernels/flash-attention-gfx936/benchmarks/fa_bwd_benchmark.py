import torch
import torch.utils.benchmark as benchmark
from collections import namedtuple
import math
import importlib.util
import csv

# 加载动态库
path_to_so = '../build/flash-attention.so'
print("load from {}".format(path_to_so))
spec = importlib.util.spec_from_file_location("flash_attn_2_cuda", path_to_so)
flash_attn_2_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flash_attn_2_cuda)
import flash_attn_2_cuda as _C_flashattention

def benchmark_backward(fn, *inputs, repeats=1, desc="", verbose=False, amp=False, amp_dtype=torch.float16, **kwinputs):
    if verbose:
        print(desc, "- Backward pass")
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose: print(m)
    return m.times[0] 

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12)

params_list = [
    {'causal': True, 'batch_size': 1, 'nheads': 16, 'nheads_k': 16, 'seq_len': 8192, 'head_size': 128, 'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 32, 'nheads_k': 32, 'seq_len': 8192, 'head_size': 128, 'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 32, 'nheads_k': 4, 'seq_len': 8192, 'head_size': 128, 'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 52, 'nheads_k': 4, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 16, 'nheads_k': 2, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 26, 'nheads_k': 2, 'seq_len': 8192, 'head_size': 128, 'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 8, 'nheads_k': 1, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 13, 'nheads_k': 1, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},

    {'causal': True, 'batch_size': 1, 'nheads': 32, 'nheads_k': 32, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 16, 'nheads_k': 16, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 8, 'nheads_k': 8, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 4, 'nheads_k': 4, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 40, 'nheads_k': 40, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},

    {'causal': True, 'batch_size': 1, 'nheads': 20, 'nheads_k': 20, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 10, 'nheads_k': 10, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 5, 'nheads_k': 5, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 32, 'nheads_k': 8, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 16, 'nheads_k': 4, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},

    {'causal': True, 'batch_size': 1, 'nheads': 8, 'nheads_k': 2, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 4, 'nheads_k': 1, 'seq_len': 8192, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 28, 'nheads_k': 4, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 14, 'nheads_k': 2, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},
    {'causal': True, 'batch_size': 1, 'nheads': 7, 'nheads_k': 1, 'seq_len': 4096, 'head_size': 128,'window_size': [-1, -1]},

]

csv_file_name = "bwd_results.csv"
fieldnames = ["batch_size", "seq_len", "head_size", "nheads", "nheads_k", "causal", "bwd_speed"]
results = []

for params in params_list:
    batch_size = params['batch_size']
    nheads = params['nheads']
    nheads_k = params['nheads_k']
    head_size = params['head_size']
    seq_len = params['seq_len']
    nheads_k = params['nheads_k']
    causal = params['causal']
    window_size_left = params['window_size'][0]
    window_size_right = params['window_size'][1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax_scale = 1.0 / math.sqrt(head_size)
    dropout_p = 0
    q = torch.randn(batch_size, nheads, seq_len, head_size, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, nheads_k, seq_len, head_size, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, nheads_k, seq_len, head_size, device=device, dtype=torch.float16, requires_grad=True)
    o = torch.randn(batch_size, nheads, seq_len, head_size, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn(batch_size, nheads, seq_len, head_size, device=device, dtype=torch.float16, requires_grad=True)
    dq = torch.empty(batch_size, nheads, seq_len, head_size, device=device, dtype=torch.float16) 
    dk = torch.empty(batch_size, nheads_k, seq_len, head_size, device=device, dtype=torch.float16) 
    dv = torch.empty(batch_size, nheads_k, seq_len, head_size, device=device, dtype=torch.float16) 
    lse = torch.randn(batch_size, nheads_k, seq_len, device=device, dtype=torch.float16) 
    input_params = (
        do,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv,
        None,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        0.0,
        False,
        None,
        None)
    fa_average_cost = 0
    # benchmark 多次取平均值
    iterations = 12
    warmup = 2
    cost_time_list = []
    for i in range(iterations):
        cost_time = benchmark_backward(_C_flashattention.bwd, *input_params, repeats=1)
        if i >= warmup:
            cost_time_list.append(cost_time)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # print(float(cost_time))
    max_cost_time = max(cost_time_list)
    cost_time_list.remove(max_cost_time)
    fa_average_cost = sum(cost_time_list) / (iterations - warmup - 1)
    calculation_amount_bwd = flops(batch_size, seq_len, head_size, nheads, causal,"bwd")
    speed_bwd = efficiency(calculation_amount_bwd, fa_average_cost)
    results.append({
    "batch_size": batch_size,
    "seq_len": seq_len,
    "head_size": head_size,
    "nheads": nheads,
    "nheads_k": nheads_k,
    "causal": causal,
    "bwd_speed": speed_bwd
})
    print("bs= {}, seq_len={}, head_size={}, nheads={}, nheads_k={}, causal={}, bwd speed={} tflops".format(batch_size, seq_len, head_size, nheads, nheads_k, causal, speed_bwd))
with open(csv_file_name, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    for result in results:
        writer.writerow(result)