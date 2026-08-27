import os
import sys
import math
import torch
import pickle
import time
import numpy
import argparse
import random
from datetime import datetime
use_cuda_toolkits = os.path.exists("/usr/local/cuda/bin/nvcc")
use_rocm_toolkits = os.path.exists("/opt/rocm/llvm/bin/clang")
use_dtk_toolkits = os.path.exists("/opt/dtk/bin/aicc")
if (use_cuda_toolkits):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
elif (use_rocm_toolkits or use_dtk_toolkits):
    try:
        from flash_attention_interface import flash_attn_varlen_func, flash_attn_2_cuda, flash_attn_with_kvcache
    except ModuleNotFoundError:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

import flash_attn_2_cuda as flash_attn_cuda

def _require_hg_varlen_symbol(name: str):
    symbol = getattr(flash_attn_cuda, name, None)
    if symbol is None:
        raise RuntimeError(
            f"{name} is unavailable in this build. Rebuild flash_attn with HAS_HG_DISPATCH enabled."
        )
    return symbol

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, do_assert=True, cos_threshold=1e-5) -> None:
    assert x.shape == y.shape, "for {}, x and y must have the same shape".format(name)
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    rel_diff_mean = (x / y).abs().mean().item()
    rel_diff_max  = (x / y).abs().max().item()
    print("name:{} cos_diff={:.12f}, RMSE=\x1b[35m{:.12f}\x1b[0m, amax_diff=\x1b[35m{:.12f}\x1b[0m, REL=\x1b[35m{:.12f}\x1b[0m, rel_max=\x1b[35m{:.12f}\x1b[0m".format(
        name, cos_diff, RMSE, amax_diff, rel_diff_mean, rel_diff_max))
    if (do_assert): assert cos_diff < cos_threshold


def scaled_dot_product_attention(__query, __key, __value, h_q, h_kv, is_causal=False, USE_CPU=False, return_max_sum=False, original_seqlen_kv=0, split_slice=0, is_bshd=False, window_size=(-1, -1)):
    __query = __query.transpose(0, 1).contiguous()
    __key = __key.transpose(0, 1).contiguous()
    __value = __value.transpose(0, 1).contiguous()
    # 判断是否使用 CPU 计算 golden, 避免 blas 的影响
    original_device = __query.device
    original_dtype  = __query.dtype
    if (USE_CPU):
        __query = __query.cpu()
        __key   = __key.cpu()
        __value = __value.cpu()
    # print("scaled_dot_product_attention: ", query.shape, key.shape, value.shape)
    __query = __query.float()
    __key   = __key.float()
    __value = __value.float()
    # 如果按照官方的方法返回
    if (not return_max_sum):
        __key   = __key.repeat_interleave(h_q // h_kv, dim=0)
        __value = __value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = __query @ __key.transpose(-2, -1) / math.sqrt(__query.size(-1))
        # MTP > 1, causal/local mask applied
        if (window_size != (-1, -1)):
            s_q = __query.shape[-2]
            s_k = __key.shape[-2]
            left, right = window_size
            if left < 0:
                left = s_k
            if right < 0:
                right = s_k
            row_idx = torch.arange(s_q, dtype=torch.int32, device=attn_weight.device)[:, None]
            col_idx = torch.arange(s_k, dtype=torch.int32, device=attn_weight.device)[None, :]
            col_idx_limit_left = row_idx + s_k - s_q - left
            col_idx_limit_right = row_idx + s_k - s_q + right
            temp_mask = (col_idx >= col_idx_limit_left) & (col_idx <= col_idx_limit_right)
            attn_weight = attn_weight.masked_fill(temp_mask.logical_not(), float("-inf"))
        elif (is_causal):
            s_q = __query.shape[-2]
            s_k = __key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=__query.dtype, device=attn_weight.device)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=attn_weight.device).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(__query.dtype)
            attn_weight += attn_bias
        # some codes for debug
        scores_max = attn_weight.to(torch.float32).max(-1)[0]
        scores_sum = torch.exp(attn_weight.to(torch.float32) - scores_max.unsqueeze(-1)).sum(dim=-1)
        # original codes
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        output = attn_weight @ __value
        output = output.transpose(0, 1).contiguous()
        return output.to(original_device).to(original_dtype), lse.to(original_device), scores_max.to(original_device), scores_sum.to(original_device)


def set_random_seed(seed=0):
    random.seed(seed)  # 设置 Python 的随机种子
    numpy.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['OMP_NUM_THREADS'] = '1'  # 设置 OpenMP 的线程数
    torch.set_num_threads(1)  # 设置 PyTorch 的线程数


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--load', default=False, action='store_true', help='load path')
    parser.add_argument('--trace', default=False, action='store_true', help='whether dump perf traces')
    parser.add_argument('--bf16', default=False, action='store_true', help='whether use bfloat16 as main dtype')
    parser.add_argument('--fp8', default=False, action='store_true', help='whether use fp8_e4m3 inputs for HG decode')
    parser.add_argument('--pressure', default=False, action='store_true', help='whether do pressure test')
    parser.add_argument('--cpu', default=False, action='store_true', help='whether compute golden via cpu')
    parser.add_argument('--pad', default=False, action='store_true', help='whether make query uncontiguous to simulate vllm behaviors')
    parser.add_argument('--iterations', type=int, default=100, help='pressure test times')
    parser.add_argument('--block_size', type=int, default=128, help='page block_size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for generated inputs')
    parser.add_argument('--seq-q', type=int, default=4, help='query length per batch for generated inputs')
    parser.add_argument('--seq-k', type=int, default=2048, help='kv length per batch for generated inputs')
    parser.add_argument('--num-heads', type=int, default=24, help='number of query heads for generated inputs')
    parser.add_argument('--num-heads-kv', type=int, default=2, help='number of kv heads for generated inputs')
    parser.add_argument('--head-dim-qk', type=int, default=128, help='query/key head dimension')
    parser.add_argument('--head-dim-v', type=int, default=128, help='value head dimension')
    parser.add_argument('--no-causal', dest='causal', default=True, action='store_false', help='disable causal mask for generated inputs')
    parser.add_argument('--window-left', type=int, default=-1, help='left sliding window size')
    parser.add_argument('--window-right', type=int, default=-1, help='right sliding window size')
    parser.add_argument('--seed', default=False, action='store_true', help='whether do pressure test')
    args = parser.parse_args()

    if (args.seed):
        set_random_seed(212)

    # 从文件加载输入
    if (args.load):
        nvidia_packet = torch.load("./demo.pt")
        query, key, value, cu_seqlens_q, max_seqlen_q, cache_seqlens, max_seqlen_k, softmax_scale, causal, window_size, alibi_slopes, page_table, softcap, fa_version, q_descale, k_descale, v_descale = nvidia_packet["inputs"]
        vllm_golden = nvidia_packet["outputs"]
        # 解析出必要的参数
        batch_size = page_table.shape[0]
        assert batch_size == cu_seqlens_q.shape[0] - 1, "check batch size"
        page_block_size = key.shape[1]
        num_heads_kv = key.shape[2]
        num_heads    = query.shape[1]
        head_dim_qk  = query.shape[2]
        head_dim_v   = key.shape[3]
        infer_dtype  = query.dtype
    else:
        # 随机生成 seqkv
        batch_size = args.batch_size

        # 得到 Q 的长度
        seqlen_q = [args.seq_q for i in range(batch_size)]
        seqlen_q_sum = sum(seqlen_q)
        max_seqlen_q = max(seqlen_q)
        cu_seqlens_q = numpy.array([0] + numpy.cumsum(seqlen_q).tolist()).astype("int32")
        cu_seqlens_q = torch.from_numpy(cu_seqlens_q)

        # 得到 KV 的长度
        cache_seqlens  = [args.seq_k for i in range(batch_size)]

        # 指定分页块的大小, nvidia 64, ours 128
        page_block_size = 16 if (use_cuda_toolkits) else args.block_size

        # 根据分页块大小计算实际需要的页表的大小
        max_seqlen_k = max(cache_seqlens)
        seqlen_kv_real_required_page = [math.ceil(it / page_block_size) for it in cache_seqlens]
        seqlen_kv_real_required_page_sum = sum(seqlen_kv_real_required_page)
        # 默认按照最大 seqlenkv 的来分配
        seqlen_kv_max_required_page = math.ceil(max_seqlen_k / page_block_size)
        seqlen_kv_max_required_page_total = batch_size * seqlen_kv_max_required_page
        # 打乱页表
        shuffle = True
        if (shuffle):
            block_random = torch.randperm(seqlen_kv_max_required_page_total, dtype=torch.int32, device="cuda")
        else:
            block_random = torch.arange(seqlen_kv_max_required_page_total , dtype=torch.int32)
        page_table    = []
        seq_block_incre = 0
        for i in range(batch_size):
            blocks_pad = [0] * seqlen_kv_max_required_page
            if (shuffle):
                blocks_pad[:seqlen_kv_real_required_page[i]] = block_random[seq_block_incre: seq_block_incre + seqlen_kv_real_required_page[i]].cpu().tolist()
                seq_block_incre += seqlen_kv_real_required_page[i]
            else:
                blocks_pad = block_random[seq_block_incre: seq_block_incre + seqlen_kv_max_required_page].cpu().tolist()
                seq_block_incre += seqlen_kv_max_required_page
            page_table.append(torch.IntTensor(blocks_pad))
        page_table = torch.stack(page_table).contiguous().to("cuda")
        # 创建基本参数
        head_dim_qk   = args.head_dim_qk
        head_dim_v    = args.head_dim_v
        num_heads     = args.num_heads
        num_heads_kv  = args.num_heads_kv
        infer_dtype   = torch.float16 # deepseek 默认使用 bfloat16 推理
        if (args.bf16): infer_dtype = torch.bfloat16 # 除非命令行指定用 fp16, 不受 args.dtype 影响
        softmax_scale = 1.0 / math.sqrt(head_dim_qk)
        causal        = args.causal
        window_size   = (args.window_left, args.window_right)
        alibi_slopes  = None
        softcap       = 0.0
        fa_version    = 2
        q_descale     = torch.ones((batch_size, num_heads), dtype=torch.float32, device="cuda")
        k_descale     = torch.ones((batch_size, num_heads_kv), dtype=torch.float32, device="cuda")
        v_descale     = torch.ones((batch_size, num_heads_kv), dtype=torch.float32, device="cuda")

        # 创建输入张量
        if (args.pad):
            query_origin_tensor = torch.randn((seqlen_q_sum, num_heads + 16, head_dim_qk), dtype=infer_dtype, device="cuda")
            q = query_origin_tensor[:, :num_heads]
        else:
            q = torch.randn((seqlen_q_sum, num_heads, head_dim_qk), dtype=infer_dtype, device="cuda")
        k_cache   = torch.randn((seqlen_kv_max_required_page_total, page_block_size, num_heads_kv, head_dim_qk), device="cuda", dtype=infer_dtype)
        v_cache = torch.randn((seqlen_kv_max_required_page_total, page_block_size, num_heads_kv, head_dim_v), device="cuda", dtype=infer_dtype)
        vllm_golden  = None
        cu_seqlens_q = cu_seqlens_q.to(q.device)
        cache_seqlens    = torch.from_numpy(numpy.array(cache_seqlens).astype("int32")).to(q.device)

    q_ref = q
    k_cache_ref = k_cache
    v_cache_ref = v_cache
    if args.fp8:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not support torch.float8_e4m3fn")
        q = q.to(torch.float8_e4m3fn)
        k_cache = k_cache.to(torch.float8_e4m3fn)
        v_cache = v_cache.to(torch.float8_e4m3fn)
        q_ref = q.to(infer_dtype)
        k_cache_ref = k_cache.to(infer_dtype)
        v_cache_ref = v_cache.to(infer_dtype)

    # 展示一下输入数据
    print("--------------------------------------------------------------------------------------------")
    print("q: ", q.shape, q.dtype, q.is_contiguous(), q.stride())
    print("k_cache: ", k_cache.shape, k_cache.dtype, k_cache.is_contiguous(), k_cache.stride())
    print("v_cache: ", v_cache.shape, v_cache.dtype, v_cache.is_contiguous(), v_cache.stride())
    print("cu_seqlens_q: ", cu_seqlens_q.shape, cu_seqlens_q.dtype, cu_seqlens_q.is_contiguous())
    print("cu_seqlens_q: ", cu_seqlens_q)
    print("max_seqlen_q: ", max_seqlen_q)
    print("cache_seqlens: ", cache_seqlens)
    print("max_seqlen_k: ", max_seqlen_k)
    print("softmax_scale: ", softmax_scale)
    print("causal: ", causal)
    print("window_size: ", window_size)
    print("alibi_slopes: ", alibi_slopes)
    print("page_table: ", page_table.shape, page_table.dtype, page_table.is_contiguous(), page_table.stride())
    print("page_table: ", page_table)
    print("softcap: ", softcap)
    print("fa_version: ", fa_version)
    print("q_descale: ", q_descale.shape, q_descale.dtype, q_descale.tolist())
    print("k_descale: ", k_descale.shape, k_descale.dtype, k_descale.tolist())
    print("v_descale: ", v_descale.shape, v_descale.dtype, v_descale.tolist())

    print("--------------------------------------------------------------------------------------------")
    # 先从 kvcache 中还原出 key 和 value
    key_original   = []
    value_original = []
    for b in range(batch_size):
        # 获取页表索引
        index = page_table[b]
        # 获取实际的索引
        max_page_blocks = math.ceil(cache_seqlens[b] / page_block_size)
        actual_index = index[:max_page_blocks]
        # 根据该页表索引获取当前 seqlenkv 的内容
        key_content = k_cache_ref[actual_index]
        # reshape 回去
        key_content = key_content.view(-1, num_heads_kv, head_dim_qk)[:cache_seqlens[b]].contiguous()
        # 同理
        value_content = v_cache_ref[actual_index].view(-1, num_heads_kv, head_dim_v)[:cache_seqlens[b]].contiguous()
        key_original.append(key_content)
        value_original.append(value_content)

    # 同理还原出 query 的内容
    query_original = []
    cum_q = 0
    for b in range(batch_size):
        query_len = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
        query_content = q_ref[cum_q: cum_q + query_len]
        query_original.append(query_content.contiguous())
        cum_q += query_len

    # 重新实现 self-attention
    golden = []
    golden_lse = []
    golden_max = []
    for b in range(batch_size):
        tmp_output, lse, scores_max, scores_sum = scaled_dot_product_attention(query_original[b], key_original[b], value_original[b], num_heads, num_heads_kv, is_causal=causal, USE_CPU=args.cpu, window_size=window_size)
        golden.append(tmp_output)
        golden_lse.append(lse)
        golden_max.append(scores_max)
    golden = torch.cat(golden, dim=0)
    golden_lse = torch.cat(golden_lse, dim=-1)
    golden_max = torch.cat(golden_max, dim=-1)
    print("golden: ", golden.shape)
    print("golden_lse: ", golden_lse.shape)
    print("--------------------------------------------------------------------------------------------")
    if (True):
        # fa_output, fa_lse = flash_attn_2_cuda.prefix_decode_varlen_fwd(
        bshd_pa_decode = _require_hg_varlen_symbol("hg_prefix_decode_varlen_fwd")
        fa_output, fa_lse = bshd_pa_decode(
            q,
            k_cache,
            v_cache,
            None, # out_
            cu_seqlens_q,
            None, # cu_seqlens_k
            cache_seqlens,
            alibi_slopes,
            page_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0, # dropout
            softmax_scale,
            False, # zero_tensors
            causal,
            window_size[0],
            window_size[1],
            softcap,
            True, # return_softmax_lse,
            1,
            q_descale if args.fp8 else None,
            k_descale if args.fp8 else None,
            v_descale if args.fp8 else None,
            None, # s_aux
            infer_dtype == torch.bfloat16,
        )
    else:
        fa_output, fa_lse, *rest = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=True,
        )
    torch.cuda.synchronize()
    if (vllm_golden is not None):
        # 检查保存流程是否有错误
        cal_diff(fa_output, vllm_golden, "check")

    print("fa_output: ", fa_output.shape)
    if (fa_lse is not None): print("fa_lse: ", fa_lse.shape)
    # 检验精度如何
    fp8_threshold = 5e-3
    cal_diff(golden, fa_output, "accuracy", True, fp8_threshold if args.fp8 else 1e-5)
    if (fa_lse is not None): cal_diff(golden_lse, fa_lse, "softmax_lse", True, fp8_threshold if args.fp8 else 1e-5)
    print("--------------------------------------------------------------------------------------------")

    # benchmark 性能数据
    import triton
    def benchmark_prefix_prefill():
        _ = bshd_pa_decode(
            q,
            k_cache,
            v_cache,
            None,
            cu_seqlens_q,
            None,
            cache_seqlens,
            alibi_slopes,
            page_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            True,
            1,
            q_descale if args.fp8 else None,
            k_descale if args.fp8 else None,
            v_descale if args.fp8 else None,
            None,
            infer_dtype == torch.bfloat16,
        )
    # 适时关闭, 用于 debug
    if ((os.getenv("FA_DEBUG") is None) and (os.getenv("HIP_LOG_LEVEL") is None) and not args.trace):
        import triton
        t = triton.testing.do_bench_cudagraph(benchmark_prefix_prefill)
        FLOPS = float(0)
        BYTES = float(0)
        for b in range(batch_size):
            batch_seqlen_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            batch_seqlen_k = cache_seqlens[b]
            effective_seqlen_k = batch_seqlen_k
            if window_size != (-1, -1):
                window_left, window_right = window_size
                left = batch_seqlen_k if window_left < 0 else window_left
                right = batch_seqlen_k if window_right < 0 else window_right
                effective_seqlen_k = min(batch_seqlen_k, left + batch_seqlen_q + right)
            undo_flops = batch_seqlen_q * batch_seqlen_q / 2 if (causal and window_size == (-1, -1)) else 0
            attn_elems = batch_seqlen_q * effective_seqlen_k - undo_flops
            qk_flops = num_heads * attn_elems * head_dim_qk * 2
            pv_flops = num_heads * attn_elems * head_dim_v * 2
            FLOPS += qk_flops + pv_flops
            q_load = batch_seqlen_q * num_heads * head_dim_qk
            k_load = effective_seqlen_k * num_heads_kv * head_dim_qk # k load not only once
            v_load = effective_seqlen_k * num_heads_kv * head_dim_v
            BYTES += q_load * q.element_size() + k_load * k_cache.element_size() + v_load * v_cache.element_size() # ignore storation ?
        print(f"Performance: {t:.3f} ms, \x1b[35m{FLOPS / 10 ** 9 / t:.2f}\x1b[0m TFLOPS, \x1b[35m{BYTES / 10 ** 6 / t:.0f}\x1b[0m GB/s")

        # 压力测试
        if (args.pressure):
            pressure_count = max(100, args.iterations)
            for p in range(pressure_count):
                pressure_fa_output = torch.zeros_like(fa_output)
                pressure_fa_output, _ = bshd_pa_decode(
                    q.clone(),
                    k_cache.clone(),
                    v_cache.clone(),
                    None,
                    cu_seqlens_q,
                    None,
                    cache_seqlens,
                    alibi_slopes,
                    page_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    0.0,
                    softmax_scale,
                    False,
                    causal,
                    window_size[0],
                    window_size[1],
                    softcap,
                    True,
                    1,
                    q_descale if args.fp8 else None,
                    k_descale if args.fp8 else None,
                    v_descale if args.fp8 else None,
                    infer_dtype == torch.bfloat16,
                )
                torch.cuda.synchronize()
                is_equal = torch.equal(pressure_fa_output, fa_output)
                if (not is_equal): cal_diff(pressure_fa_output, fa_output, "pressure")
                assert is_equal, "\x1b[31mUnstable\x1b[0m!"
                del pressure_fa_output
                sys.stdout.write("\rPressure Test: {}/{}".format(p + 1, pressure_count))
            print(" \x1b[32mPASS\x1b[0m")
        print("-----------------------------------------------------------------------------------")
