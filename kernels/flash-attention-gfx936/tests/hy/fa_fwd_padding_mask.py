import os
import sys
import math
import time
import numpy
import random
import torch
import site
import importlib
import flash_attn
from einops import rearrange
import torch.utils.benchmark as benchmark
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from flash_attn.flash_attn_interface import flash_attn_cuda as flash_attn_2_cuda
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import pdb

# 加载动态库
# path_to_so = "../build/libflash_attention.so"
path_to_so = os.path.join(site.getsitepackages()[0], 'flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so')
print("load from {}".format(path_to_so))
spec = importlib.util.spec_from_file_location("flash_attn_2_cuda", path_to_so)
flash_attn_2_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flash_attn_2_cuda)

def benchmark_forward(fn, *inputs, repeats=100, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m

# 辅助打印
def get_status_str(pass_status):
    return "\x1b[32mPASS\x1b[0m" if (pass_status) else "\x1b[31mFAIL\x1b[0m"

# 获取 flops
def get_fa_flops(batch, seq_len, qheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    if (isinstance(seq_len, int)):
        f = 4 * batch * seq_len**2 * qheads * headdim // (2 if causal else 1)
    else:
        tmp = sum([(seq_len[k + 1] - seq_len[k])**2 for k in range(len(seq_len) - 1)])
        f = 4 * tmp * qheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

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
    
    def __float__(self):
        return self.interval * 1e3


def standard_attention(_Q, _K, _V, causal_mask=False, i=0, do_print=False, softmax_type=torch.float32, is_local=False, use_alibi=False):
    original_device = _Q.device
    original_dtype  = _Q.device
    USE_CPU = bool(os.getenv("USE_CUDA") is None)
    if (USE_CPU):
        _Q = _Q.cpu().to(torch.float32)
        _K = _K.cpu().to(torch.float32)
        _V = _V.cpu().to(torch.float32)
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


random.seed(0)
seqlen_qkv   = None
num_heads    = 12
num_heads_kv = 12
head_dim_qk  = 64
head_dim_v   = 64
causal_mask  = False
MAX_SEQLEN_KV = 384
if (seqlen_qkv is None):
    # batch_size = random.randint(1, 100)
    batch_size = 80
    seqlen_qkv = [random.randint(1, MAX_SEQLEN_KV) for i in range(batch_size)]
else:
    batch_size = len(seqlen_qkv)
seqlen_qkv_sum = sum(seqlen_qkv)
seqlen_qkv_max = max(seqlen_qkv)
prefill_meta_seq_start_loc = numpy.array([0] + numpy.cumsum(seqlen_qkv).tolist()).astype("int32")
prefill_meta_seq_start_loc = torch.from_numpy(prefill_meta_seq_start_loc).cuda()
# 随机生成输入 query, key, value, 原生 bhsd layout
query = torch.randn((seqlen_qkv_sum, num_heads, head_dim_qk), dtype=torch.float16, device="cuda")
key   = torch.randn((seqlen_qkv_sum, num_heads_kv, head_dim_qk), dtype=torch.float16, device="cuda")
value = torch.randn((seqlen_qkv_sum, num_heads_kv, head_dim_v), dtype=torch.float16, device="cuda")
print("-------------------------------------------------")
print("(generating inputs/golden from scratch...)")
print("query: {}\nkey: {}\nvalue: {}".format(query.shape, key.shape, value.shape))
print("seqlen_kv       :", seqlen_qkv)
print("batch_size      :", batch_size)
print("seqlen_qkv_sum  :", seqlen_qkv_sum)
print("seqlen_qkv_max  :", seqlen_qkv_max)
# 拆分出每个 batch 的结果
query_batch = []
key_batch   = []
value_batch = []
golden_batch = []
for i in range(batch_size):
    # 从 batch x num_heads x seqlen, head_dim 中解析出每个 batch 的 Q/K/V 内容
    query_batch.append(query[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
    key_batch.append(key[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
    value_batch.append(value[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
    # 计算 golden
    golden_now = standard_attention(query_batch[-1], key_batch[-1], value_batch[-1], causal_mask=causal_mask)[0]
    golden_batch.append(golden_now)
    sys.stdout.write("\rgolden computing: {}/{} -- {}".format(i + 1, batch_size, golden_now.shape))
    sys.stdout.flush()
print("")
# golden 输出应该和 query 一样大
golden = []
for i in range(batch_size):
    # print(golden_batch[i].shape)
    padding_mask = torch.zeros((1, num_heads, 384, head_dim_v), dtype=query.dtype, device=query.device)
    padding_mask[:, :, :seqlen_qkv[i]] = golden_batch[i]
    golden.append(padding_mask)
golden = torch.cat(golden, dim=0).transpose(1, 2).contiguous()
print("golden: ", golden.shape)
if (os.getenv("FA_DEBUG") is not None):
    print("-------------------- args ------------------------")
    print("query: ", query.shape, query.dtype)
    print("key: ", key.shape, key.dtype)
    print("value: ", value.shape, value.dtype)
    print("out_: ", None)
    print("cu_seqlens_q: ", prefill_meta_seq_start_loc)
    print("cu_seqlens_kv: ", prefill_meta_seq_start_loc)
    print("seqused_k: ", None)
    print("alibi_slopes: ", None)
    print("seqlen_q_max: ", seqlen_qkv_max)
    print("seqlen_kv_max: ", seqlen_qkv_max)
    print("p_dropout: ", 0.0)
    print("softmax_scale: ", 1.0 / math.sqrt(head_dim_qk))
    print("zero_tensors: ", None)
    print("is_causal: ", True)
    print("window_size_left: ", -1)
    print("window_size_right: ", -1)
    print("softcap: ", 0.0)
    print("return_softmax: ", False)
    print("gen_: ", None)
    print("-------------------------------------------------")
else:
    print("-------------------------------------------------")
# 对 Q, K, V 做 padding
query_padded = torch.zeros((batch_size, 384, num_heads, head_dim_qk), dtype=query.dtype, device=query.device)
key_padded   = torch.zeros((batch_size, 384, num_heads_kv, head_dim_qk), dtype=query.dtype, device=query.device)
value_padded = torch.zeros((batch_size, 384, num_heads_kv, head_dim_v), dtype=query.dtype, device=query.device)
for i in range(batch_size):
    query_padded[i, :seqlen_qkv[i]] = query_batch[i].transpose(1, 2).contiguous()
    key_padded[i, :seqlen_qkv[i]]   = key_batch[i].transpose(1, 2).contiguous()
    value_padded[i, :seqlen_qkv[i]] = value_batch[i].transpose(1, 2).contiguous()
query_padded = query_padded.contiguous()
key_padded   = key_padded.contiguous()
value_padded = value_padded.contiguous()
print("query_padded: ", query_padded.shape)
print("key_padded  : ", key_padded.shape)
print("value_padded: ", value_padded.shape)
padding_mask = torch.tensor(seqlen_qkv).to(torch.int32).to(query.device)
print("padding_mask: ", padding_mask)
fa_output = flash_attn_2_cuda.fwd_padding_mask(
    query_padded,
    key_padded,
    value_padded,
    padding_mask,
    None,
    None,
    0.0,
    1.0 / math.sqrt(head_dim_qk),
    causal_mask,
    -1,
    -1,
    0.0,
    False,
    None,
    False
)[0]

print("fa_output: ", fa_output.shape)
# 比较差异
abs_diff = torch.abs(fa_output - golden)
rel_diff = torch.abs(fa_output / golden)
mean_abs_diff = abs_diff.mean()
max_abs_diff  = abs_diff.max()
mean_rel_diff = rel_diff.mean()
# print("golden:", golden)
# print("fa_output:", fa_output)
# torch.nonzero((golden - fa_output).abs()>0.01)
# pdb.set_trace()
print("abs=\x1b[35m{:.12f}\x1b[0m  |  amax=\x1b[35m{:.12f}\x1b[0m".format(mean_abs_diff, max_abs_diff))

# 检查是否有异常值
occur_nan_in_output = torch.any(torch.isnan(fa_output))
occur_inf_in_output = torch.any(torch.isinf(fa_output))
print("CHECK NaN: ", get_status_str(not occur_nan_in_output))
print("CHECK INF: ", get_status_str(not occur_inf_in_output))

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("prof_data/test_trace_" + str(prof.step_num) + ".json")
# 测试性能
if (os.getenv("FA_DEBUG") is None):
    cost_time = []
    for __iter in range(10):
        t = benchmark_forward(flash_attn_2_cuda.fwd_padding_mask, query_padded, key_padded, value_padded, padding_mask, None, None, 0.0, 1.0 / math.sqrt(head_dim_qk), causal_mask, -1, -1, 0.0, False, None, False, repeats=1, verbose=False)[1].times[0]
        if (__iter > 0): cost_time.append(t)
    cost_time = numpy.array(cost_time)
    cost_time_mean = cost_time.mean()
    cost_time = numpy.delete(cost_time, numpy.where(cost_time < (0.8 * cost_time_mean))) # 去除突发低数据
    cost_time_mean = cost_time.mean()
    # 统计计算量
    batch_size = query_padded.shape[0]
    seq_len    = query_padded.shape[1]
    num_heads  = query_padded.shape[2]
    head_dim   = query_padded.shape[3]
    flops_count = get_fa_flops(batch_size, seq_len, num_heads, head_dim, True, "fwd")
    tflops = flops_count / (cost_time_mean * 1e12)
    print("Performance: \x1b[35m{:.1f}\x1b[0m TFLOPS, {:.4f} ms".format(tflops, cost_time_mean * 1e3))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=10),
        on_trace_ready=trace_handler
    ) as prof:
        # 压力测试
        pressure_count = 100
        for p in range(pressure_count):
            torch.cuda.empty_cache()
            pressure_query_layer = query_padded.clone()
            pressure_key_layer   = key_padded.clone()
            pressure_value_layer = value_padded.clone()
            outputs = flash_attn_2_cuda.fwd_padding_mask(
                pressure_query_layer,
                pressure_key_layer,
                pressure_value_layer,
                padding_mask,
                None,
                None,
                0.0,
                1.0 / math.sqrt(head_dim_qk),
                causal_mask,
                -1,
                -1,
                0.0,
                False,
                None,
                False
            )
            torch.cuda.synchronize()
            pressure_fa_output   = outputs[0]
            # print("pressure_fa_output:", pressure_fa_output)
            # print("fa_output:", fa_output)
            assert torch.equal(pressure_fa_output, fa_output), "Unstable"
            pressure_query_layer.fill_(0)
            pressure_key_layer.fill_(0)
            pressure_value_layer.fill_(0)
            del pressure_query_layer, pressure_key_layer, pressure_value_layer
            sys.stdout.write("\rPressure Test: {}/{}".format(p + 1, pressure_count))
            prof.step()
    print(" \x1b[32mPASS\x1b[0m")