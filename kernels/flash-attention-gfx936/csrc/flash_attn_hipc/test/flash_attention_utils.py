#!/usr/bin/env python
from typing import Optional, Union
import os
import sys
import importlib.util
import argparse
import random
import time
import math
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torch.utils.benchmark as benchmark
from einops import rearrange, repeat


# 辅助打印
def get_status_str(pass_status):
    return "\x1b[32mPASS\x1b[0m" if (pass_status) else "\x1b[31mFAIL\x1b[0m"

# 计算 tflops
def efficiency(flop, time):
    return (flop / time / 10**12)

# 计时器
class Timer:
    def __init__(self, do_print=False):
        self.print = do_print

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end = time.process_time()
        self.interval = self.end - self.start
        if (self.print): print("{:.4f} ms".format(self.interval * 1e3)) # ms

    def __format__(self, code):
        return "{:.2f}".format(self.interval * 1e6) # us

    def __float__(self):
        return self.interval * 1e6


# 计时器
def benchmark_forward(fn, *inputs, repeats=1, desc="", verbose=False, amp=False, amp_dtype=torch.float16, **kwinputs):
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
    if verbose: print(m)
    return m.times[0] * 1e6 # return us


# 获取 flops
def get_fa_flops(batch, seq_len, qheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if (isinstance(seq_len, int)):
        f = 4 * batch * seq_len**2 * qheads * headdim // (2 if causal else 1)
    else:
        tmp = sum([(seq_len[k + 1] - seq_len[k])**2 for k in range(len(seq_len) - 1)])
        f = 4 * tmp * qheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

# 获取 cross attention flops
def get_cross_fa_flops(batch, seqlen_q, seqlen_kv, qheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if (isinstance(seqlen_q, int)):
        qk_gemm_flops = 2 * batch * qheads * (seqlen_q * seqlen_kv * headdim)
        pv_gemm_flops = 2 * batch * qheads * (seqlen_q * headdim * seqlen_kv)
        f = (qk_gemm_flops + pv_gemm_flops) // (2 if causal else 1)
    else:
        tmp = sum([(seqlen_q[k + 1] - seqlen_q[k])**2 for k in range(len(seqlen_q) - 1)])
        f = 4 * tmp * qheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


# 切换 bhsd, bshd
def warp_tensor(tensor, gpu_is_ours, is_varlen=False, num_head=None):
    if (not is_varlen):
        return tensor if (gpu_is_ours) else tensor.transpose(1, 2).contiguous()
    else:
        return tensor if (gpu_is_ours) else tensor.view(-1, num_head, tensor.shape[-1])


def per_token_quantization(input_tensor: torch.Tensor):
    """
    per-token量化
    
    Args:
        input_tensor: 输入张量 (支持形状 [seq_len, hidden_dim], 
            [batch, seq_len, hidden_dim], 或 [batch, num_heads, seq_len, head_dim])
        
    Returns:
        quantized: INT8张量, 形状与输入相同
        scales: FP32缩放因子, 形状为输入形状的前N-1维
    """
    # 保留原始维度信息
    orig_shape = input_tensor.shape
    x = input_tensor.float()  # 统一转换为FP32处理
    
    # 计算绝对最大值（最后一个维度）
    max_vals = torch.amax(torch.abs(x), dim=-1, keepdim=True)  # 形状: [..., 1]
    
    # 计算缩放因子（保持数值稳定性）
    eps = 1e-7
    scales = (max_vals.clamp(min=eps)) / 127.0  # 形状: [..., 1]
    
    # 量化计算（保持维度广播）
    quantized = (x / scales).round().clamp(-128, 127).to(torch.int8)
    
    # 恢复原始形状并处理缩放因子
    scales = scales.squeeze(-1) # 移除最后一个维度
    
    return quantized, scales


def set_random_seed(seed=0):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['OMP_NUM_THREADS'] = '1'  # 设置 OpenMP 的线程数
    torch.set_num_threads(1)  # 设置 PyTorch 的线程数


def get_partition(batch_size, mtp, max_seqlen_k, nheads_q, nheads_k, qk_head_size, v_head_size, input_dtype, input_device, device_cu=100):
    # 如果用 C++ 层的 splitkv(内存管理会做的更好), 直接返回 0
    if (os.getenv("PA_NO_SPLITKV") is not None or mtp > 1):
        return 0, None, None
    # 计算一下划分大小和划分策略
    # 如果初步能划分的 block 数量对应的利用率不高, 或者 seqKV 足够长
    partition_size = 0
    scores_raw     = None
    tmp_output     = None
    threshold      = device_cu * 0.75
    n_group        = int(nheads_q / nheads_k)
    # 如果 gqa 组数不是常见的 16/8/4/2/9/7/5/3 的倍数, ngroup 会被 re-group 到 seqlen 维度上, 会导致发的 TG 比较少
    use_regroup = all(n_group % it != 0 for it in [29, 16, 8, 4, 2, 9, 7, 5, 3])
    if (use_regroup): n_group = 1
    # 如果目前能发的 TG 数量比较少而且最大的 seqkv 不是很短
    # 或者 seqkv 比较长, 可以做切分
    if ((batch_size * mtp * n_group < threshold and max_seqlen_k >= 1024) or (max_seqlen_k >= 8192)):
        # 根据最大的 seqKV 长度, 决定相应的划分 size
        if (max_seqlen_k <= 1024): partition_size = 128
        elif (max_seqlen_k <= 2048): partition_size = 256
        elif (max_seqlen_k <= 32768): partition_size = 512
        else: partition_size = 1024
        # 如果是 MHA, 无法做 GQA ngroup-swapped 优化, 可以发更多的 TG, 不需要划分那么多小块
        if (nheads_q == nheads_k): partition_size = 1024
        # 如果按照上述划分之后, 利用率还不是很高, partition size 继续减半
        while ((nheads_q > nheads_k) and (batch_size * mtp * n_group * (max_seqlen_k / partition_size)) < threshold):
            # 目前支持的最小 partition size 是 128
            if (partition_size < 256): break
            partition_size = int(partition_size / 2)
        # 检查环境变量
        partition_size_env = os.getenv("PA_PARTITION_SIZE")
        if (partition_size_env is not None):
            partition_size = int(partition_size_env)
        # 申请 scores_max/sum 和 out_accum 的空间
        num_splits = max(1, math.floor(max_seqlen_k * 1.0 / partition_size))
        if (num_splits > 1024): return 0, None, None
        scores_raw = torch.empty(
            size=(2, num_splits, batch_size, nheads_q),
            dtype=torch.float32,
            device=input_device
        )
        tmp_output = torch.empty(
            size=(num_splits, batch_size, nheads_q, v_head_size),
            dtype=input_dtype,
            device=input_device
        )
    return partition_size, scores_raw, tmp_output


# 原始 attention 常用 golden 计算方法
def standard_attention(_Q, _K, _V, causal_mask=False, i=0, do_print=False, softmax_type=torch.float32, is_local=False, use_alibi=False):
    original_device = _Q.device
    original_dtype  = _Q.device
    USE_CPU = bool(os.getenv("USE_CUDA") is None)
    if (USE_CPU):
        _Q = _Q.cpu().to(torch.float32)
        _K = _K.cpu().to(torch.float32)
        _V = _V.cpu().to(torch.float32)
    _K = _K.repeat_interleave(_Q.shape[1] // _K.shape[1], dim=1)
    _V = _V.repeat_interleave(_Q.shape[1] // _V.shape[1], dim=1)
    # _Q * K^T
    input_dtype = _Q.dtype
    input_headdim = _Q.shape[-1]
    S = torch.matmul(_Q, _K.transpose(2, 3))
    if (do_print):
        print("==>group ", i)
        print("QK: ", S.shape, S.dtype)
    S = S.type(softmax_type)
    if (causal_mask or is_local):
        try:
            from numba import jit
        except:
            os.system("pip3 install numba")
            from numba import jit
        @jit(nopython=True)
        def generate_mask(n):
            arr = numpy.zeros((n, n), dtype="int")
            for i in range(n):
                for j in range(n):
                    if ((not is_local and j > i) or (is_local and (i > j + window_size_left or i < j - window_size_right))):
                        arr[i, j] = 0
                    else:
                        arr[i, j] = 1
            return arr
        seq_len = _Q.shape[2]
        mask = generate_mask(seq_len)
        mask = torch.from_numpy(mask).to(_Q.device)
        mask = mask.repeat(_Q.shape[0], _Q.shape[1], 1, 1)
        S = torch.where(mask > 0.5, S, -torch.inf)
    # do Alibi encoding
    _alibi_slope = None
    if (use_alibi):
        _alibi_slope  = get_alibi_slope(S.shape[1])
        relative_pos = get_relative_positions(seq_len)
        bias = _alibi_slope * relative_pos
        bias = bias.unsqueeze(0).repeat(_Q.shape[0], 1, 1, 1).to(_Q.device)
        # attention! add bias before scale, and thus, bias multiply scale
        S = S + bias * math.sqrt(input_headdim * 1.0)
        _alibi_slope = _alibi_slope.unsqueeze(0).repeat(_Q.shape[0], 1, 1, 1).to(_Q.device)
    # P = softmax(S / 根号 head_dim)
    scale_softmax = 1.0 / math.sqrt(input_headdim * 1.0)
    S_scaled = S.type(softmax_type) * scale_softmax
    # P = torch.softmax(S_scaled, dim=-1).type(input_dtype)
    S_m, S_idx = torch.max(S_scaled, dim=-1, keepdim=True)
    S_l = torch.exp(S_scaled - S_m).sum(dim=-1, keepdim=True)
    _lse = S_m + torch.log(S_l)
    P = (torch.exp(S_scaled - S_m) / S_l).type(input_dtype)
    if (do_print): print("P : ", P.shape, P.dtype)
    # O = P * V
    _O = torch.matmul(P, _V)
    # print("_O: ", _O)
    return _O.to(original_device).to(original_dtype), _lse, _alibi_slope, S_m / scale_softmax , S_l


# Attention 变体 (PA) 常用 golden 计算方法
def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False, USE_CPU=False, return_max_sum=False, original_seqlen_kv=0, split_slice=0, is_bshd=False, transpose_io=False):
    # 判断是否使用 CPU 计算 golden, 避免 blas 的影响
    original_device = query.device
    original_dtype  = query.dtype
    if (USE_CPU):
        query = query.cpu()
        key   = key.cpu()
        value = value.cpu()
    if (transpose_io):
        query = query.transpose(0, 1).contiguous()
        key = key.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()
    # print("scaled_dot_product_attention: ", query.shape, key.shape, value.shape)
    query = query.float()
    key   = key.float()
    value = value.float()
    # 如果按照官方的方法返回
    if (not return_max_sum):
        key   = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        # MTP > 1, causal mask applied
        if (is_causal):
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=attn_weight.device)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=attn_weight.device).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        # some codes for debug
        scores_max = attn_weight.to(torch.float32).max(-1)[0]
        scores_sum = torch.exp(attn_weight.to(torch.float32) - scores_max.unsqueeze(-1)).sum(dim=-1)
        # original codes
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ value
        if (transpose_io): output = output.transpose(0, 1).contiguous()
        return output.to(original_device).to(original_dtype), lse.to(original_device), scores_max.to(original_device), scores_sum.to(original_device)
    # 按照自己需要的格式返回
    else:
        softmax_scale   = 1.0 / math.sqrt(query.shape[-1])
        qkt     = query @ key.transpose(-2, -1)
        if (is_causal):
            s_q = query.shape[-3] if (is_bshd) else query.shape[-2]
            s_k = original_seqlen_kv
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=original_seqlen_kv - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_bias = attn_bias[:, split_slice]
            if (is_bshd): attn_bias = attn_bias.unsqueeze(1)
            qkt += attn_bias
        qkt_max = qkt.max(dim=-1)[0].unsqueeze(-1)
        qkt_exp = torch.exp((qkt - qkt_max) * softmax_scale)
        qkt_sum = qkt_exp.sum(-1).unsqueeze(-1)
        qkt_softmax = qkt_exp / qkt_sum
        pv = torch.matmul(qkt_softmax, value)
        return pv.to(original_device).to(original_dtype), None, (qkt_max * softmax_scale).to(original_device).squeeze(-1), qkt_sum.to(original_device).squeeze(-1)


# 计算精度
def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, do_assert=True) -> None:
    assert x.shape == y.shape, "for {}, x and y must have the same shape".format(name)
    if (x.device != y.device):
        x, y = x.cpu(), y.cpu()
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    rel_diff_mean = (x / y).abs().mean().item()
    rel_diff_max  = (x / y).abs().max().item()
    print("name:{} cos_diff={:.12f}, RMSE=\x1b[35m{:.12f}\x1b[0m, amax_diff=\x1b[35m{:.12f}\x1b[0m, REL=\x1b[35m{:.12f}\x1b[0m, rel_max=\x1b[35m{:.12f}\x1b[0m".format(
        name, cos_diff, RMSE, amax_diff, rel_diff_mean, rel_diff_max))
    if (do_assert): assert cos_diff < 1e-5


try:
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b','--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-nq','--num_heads_q', type=int, default=16, help='Number of heads')
    parser.add_argument('-nk','--num_heads_k', type=int, default=2, help='Number of heads')
    parser.add_argument('-d','--qk_head_size', type=int, default=128, help='Size of each qk head')
    parser.add_argument('-dv','--v_head_size', type=int, default=128, help='Size of each v head')
    parser.add_argument('--dropout_p', type=int, default=0, help='dropout')
    parser.add_argument('--softmax_scale', type=int, default=1, help='softmax_scale')
    parser.add_argument('--block_size', type=int, default=128, help='page table block size')
    parser.add_argument('--max_seqlen', type=int, default=16, help='max seqlen for KV')
    parser.add_argument('--capbility', type=int, default=3, help='dtype for inference')
    parser.add_argument('--warmup', type=int, default=2, help='warmup times')
    parser.add_argument('--repeats', type=int, default=1, help='run times during once benchmark')
    parser.add_argument('--iterations', type=int, default=10, help='iterations for benchmarking')
    parser.add_argument('--mtp', type=int, default=1, help='multi token prediction count, default: 1')
    parser.add_argument('-s','--seq_len', type=str, default="(8192)", help='Sequence length')
    parser.add_argument('--device', type=str, default="cuda", help='Device of tensor')
    parser.add_argument('--dtype', type=str, default="fp16", help='dtype for inference')
    parser.add_argument('--dump', type=str, default=None, help='whether save input and output of PA')
    parser.add_argument('--load', type=str, default=None, help='input and results of pre-dumped pth')
    parser.add_argument('--layout', type=str, default="bhsd", help='decode which layout to use, optional: bhsd, bshd, sbhd')
    parser.add_argument('--compare', type=str, default=None, help='competitor card name')
    parser.add_argument('--causal', type=bool, default=False, help='causal')
    parser.add_argument('--bf16', default=False, action='store_true', help='whether use bfloat16 as main dtype')
    parser.add_argument('--golden', default=False, action='store_true', help='whether compare results with golden')
    parser.add_argument('--ali', default=False, action='store_true', help='whether use ali size only')
    parser.add_argument('--ratio', default=False, action='store_true', help='whether compute ratio of ours card/nvidia')
    parser.add_argument('--ratio_prior', default=False, action='store_true', help='whether compute ratio of ours card/prior card')
    parser.add_argument('--trace', default=False, action='store_true', help='when trace is on, precision will not be verified')
    parser.add_argument('--seed', default=False, action='store_true', help='whether fix random number seed to reproduce')
    parser.add_argument('--cpu', default=False, action='store_true', help='whether compute golden via cpu rather than gpu')
    parser.add_argument('--whl', default=False, action='store_true', help='whether test using flash_attn in whl rather than in temporary .so')
    parser.add_argument('--mla', default=False, action='store_true', help='whether test with flashMLA decoding')
    parser.add_argument('--random', default=False, action='store_true', help='whether generate random seqlen_kv for testing')
    parser.add_argument('--splitkv', default=False, action='store_true', help='whether test with splitkv-flashMLA')
    parser.add_argument('--fp16', default=False, action='store_true', help='whether test flashMLA with float16 precision rather than default bfloat16 precision')
    parser.add_argument('--detail', default=False, action='store_true', help='whether output more detailed information for debugging')
    parser.add_argument('--pressure', default=False, action='store_true', help='whether do pressure test')
    parser.add_argument('--nograph', default=False, action='store_true', help='whether test with cuda_graph assumption')
    parser.add_argument('--triton', default=False, action='store_true', help='whether test triton version flashMLA')
    parser.add_argument('--pad', default=False, action='store_true', help='whether make query uncontiguous to simulate vllm behaviors')
    parser.add_argument('--table', default=False, action='store_true', help='whether display results by table')


    args = parser.parse_args()
    args.seq_len = eval(args.seq_len)

    if (args.seed):
        set_random_seed(212)

    # 是否测 cuda-graph 模式
    if (args.nograph):
        setattr(args, "graph", False)
    else:
        setattr(args, "graph", True)

    # 判断环境
    use_cuda_toolkits = os.path.exists("/usr/local/cuda/bin/nvcc")
    use_rocm_toolkits = os.path.exists("/opt/rocm/llvm/bin/clang") or os.path.exists("/opt/dtk")

    # 加载动态库
    if (use_rocm_toolkits and not args.whl):
        path_to_so = '../../../build/flash_attn_hg/libflash_attention.so'
        if (not os.path.exists(path_to_so)):
            path_to_so = '../../build/libflash_attention.so'
        print("\x1b[33mload from {}\x1b[0m".format(path_to_so))
        spec = importlib.util.spec_from_file_location("flash_attn_hg_cuda", path_to_so)
        flash_attn_hg_cuda = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flash_attn_hg_cuda)
    else:
        import flash_attn_2_cuda as flash_attn_hg_cuda
        print("\x1b[33mload from {}\x1b[0m".format(flash_attn_2_cuda.__file__))
    # 别名
    flash_attn_cuda = flash_attn_hg_cuda

except:
    print("argparse.ArgumentParser disabled")
