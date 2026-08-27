#!/usr/bin/env python
# Benchmark: 不同 size 下 FlashAttention 无 attnmask vs 有 attnmask 的延时与速度比。
#
# 直接运行（无参数）一次性输出 4 张表：fwd causal=True、fwd causal=False、bwd causal=True、bwd causal=False
#   python benchmarks/benchmark_attnmask.py
# 仅 forward：python benchmarks/benchmark_attnmask.py --no-backward
# 仅 causal=True：python benchmarks/benchmark_attnmask.py --no-causal --causal  （或只 --no-both-causal）
# 详细对比（非表格）：python benchmarks/benchmark_attnmask.py --no-table

import argparse
import sys

# 需要与常见 benchmark 表格同尺寸时，可传：--sizes "1,1024 1,2048 1,4096 1,8192 1,16384 1,32768 8,1024 ..."
import math
import torch

from flash_attn import flash_attn_func, flash_attn_with_mask_func
from flash_attn.utils.benchmark import benchmark_forward, benchmark_fwd_bwd


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    """FLOPs 与 benchmark_flash_attention.py / fa_bwd_benchmark.py 一致。
    fwd: 4*B*S²*H*d // (2 if causal else 1)；bwd: 2.5*f；fwd_bwd: 3.5*f。
    """
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time_sec):
    """TFLOPs/s = flop / time_sec / 1e12，与 benchmark_flash_attention / fa_bwd_benchmark 一致。"""
    return (flop / time_sec / 10**12) if not math.isnan(time_sec) and time_sec > 0 else 0.0


def attn_mask_bytes(batch, nheads_q, seqlen):
    """attn_mask (batch, nheads_q, seqlen, seqlen) bool 的字节数。"""
    return batch * nheads_q * seqlen * seqlen  # 1 byte per bool


def _time_forward_ms(fn, *args, repeats=30, **kwargs):
    _, m = benchmark_forward(fn, *args, repeats=repeats, verbose=False, **kwargs)
    return m.mean * 1000.0


def _time_fwd_bwd_ms(fn, *args, repeats=30, **kwargs):
    (_, m_fwd), (_, m_bwd) = benchmark_fwd_bwd(fn, *args, repeats=repeats, verbose=False, **kwargs)
    return m_fwd.mean * 1000.0, m_bwd.mean * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark: 无 attnmask vs 有 attnmask 延时与速度比。默认直接打表。")
    parser.add_argument("--table", action="store_true", default=True, help="打印表格（默认开启）")
    parser.add_argument("--no-table", action="store_false", dest="table", help="不打印表格，改为详细对比格式")
    parser.add_argument("--batch", type=int, nargs="+", default=[128], help="batch sizes（未指定 --sizes 时）")
    parser.add_argument("--seqlen", type=int, nargs="+", default=[512, 1024, 1280, 1536, 2048], help="sequence lengths（未指定 --sizes 时）")
    parser.add_argument("--sizes", type=str, default=None, help="(batch,seqlen) 对，空格分隔；不传则用 --batch 与 --seqlen 的笛卡尔积")
    parser.add_argument("--nheads", type=int, default=28, help="nheads_q 默认值（未指定 --nheads-q 时）")
    parser.add_argument("--nheads-q", type=int, default=None, help="query 头数，默认 28")
    parser.add_argument("--num-heads-kv", type=int, default=4, help="kv 头数，默认 4（GQA）")
    parser.add_argument("--headdim", type=int, nargs="+", default=[64, 128], help="head 维度，默认 64,128")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--causal", action="store_true", default=True, help="causal=True（默认）")
    parser.add_argument("--no-causal", action="store_false", dest="causal", help="causal=False")
    parser.add_argument("--both-causal", action="store_true", default=True, help="同时跑 causal True 与 False（默认开启，无参时出 4 张表）")
    parser.add_argument("--no-both-causal", action="store_false", dest="both_causal", help="只跑当前 --causal 一种")
    parser.add_argument("--backward", action="store_true", default=True, help="是否测 backward（默认开启，无参时出 4 张表）")
    parser.add_argument("--no-backward", action="store_false", dest="backward")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--max-mask-gb", type=float, default=24.0, help="attn_mask 显存超过此值(GiB)时跳过该尺寸，避免 OOM；0 表示不限制")
    args = parser.parse_args()

    nheads_q = args.nheads_q if args.nheads_q is not None else args.nheads
    num_heads_kv = args.num_heads_kv if args.num_heads_kv is not None else nheads_q
    assert nheads_q % num_heads_kv == 0, "nheads_q must be divisible by num_heads_kv (GQA)"

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_str = "float16" if args.dtype == "fp16" else "bfloat16"
    if args.sizes:
        batch_sizes, seqlens = [], []
        for pair in args.sizes.split():
            b, s = pair.split(",")
            batch_sizes.append(int(b))
            seqlens.append(int(s))
        size_pairs = list(zip(batch_sizes, seqlens))
    else:
        size_pairs = None
        batch_sizes = args.batch
        seqlens = args.seqlen
    headdims = args.headdim
    repeats = args.repeats
    causal_vals = [True, False] if args.both_causal else [args.causal]
    fwd_header = "batch_size\tseqlen\tseqlen\tnheads_q\tnum_heads_kv\tcausal\tdim\tdtype\ttflops_attnmask_fwd\ttime_attnmask_fwd(ms)\ttflops_no_fwd\ttime_no_fwd(ms)\tfwd(%)"
    bwd_header = "batch_size\tseqlen\tseqlen\tnheads_q\tnum_heads_kv\tcausal\tdim\tdtype\ttflops_attnmask_bwd\ttime_attnmask_bwd(ms)\ttflops_no_bwd\ttime_no_bwd(ms)\tbwd(%)"

    for headdim in headdims:
        run_bwd = args.backward and headdim in (64, 128)
        if args.table:
            print(f"\n=== dim={headdim} ===", flush=True)
        for causal in causal_vals:
            rows_bwd = []
            if args.table:
                if run_bwd:
                    print(fwd_header, flush=True)
                else:
                    print("batch_size\tseqlen\tseqlen\tnheads_q\tnum_heads_kv\tcausal\tdim\tdtype\ttflops_attnmask\ttime_attnmask(ms)\ttflops_no\ttime_no(ms)\ttflops_attnmask/no_attnmask(%)", flush=True)
            else:
                print("\n" + "=" * 90)
                print("Benchmark: 无 attnmask vs 有 attnmask — 各 size 延时 (ms) 与速度比 (attnmask/no_attnmask)")
                print("=" * 90)
                print(f"  dtype={args.dtype}, nheads_q={nheads_q}, num_heads_kv={num_heads_kv}, headdim={headdim}, causal={causal}, repeats={repeats}")
                if args.backward and headdim not in (64, 128):
                    print("  backward 对比仅在 headdim=64/128 时执行，当前 dim 只统计 forward。")
                if run_bwd:
                    print(f"  {'batch':>5} {'seqlen':>7} │ {'no_attnmask_fwd':>12} {'attnmask_fwd':>12} {'ratio_fwd':>9} │ "
                          f"{'no_attnmask_bwd':>12} {'attnmask_bwd':>12} {'ratio_bwd':>9}")
                else:
                    print(f"  {'batch':>5} {'seqlen':>7} │ {'no_attnmask(ms)':>14} {'attnmask(ms)':>14} │ {'speed_ratio':>10}  (attnmask/no_attnmask, >1 表示 attnmask 更慢)")
                print("-" * 90)

            for batch, seqlen in (size_pairs if size_pairs else ((b, s) for b in batch_sizes for s in seqlens)):
                    mask_gb = attn_mask_bytes(batch, nheads_q, seqlen) / (1024**3)
                    if args.max_mask_gb > 0 and mask_gb > args.max_mask_gb:
                        if args.table:
                            skip_row = f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\t-\t-\tskip(OOM)\t{mask_gb:.1f}GiB_mask\t-"
                            print(skip_row, flush=True)
                            if run_bwd:
                                rows_bwd.append(skip_row)
                        else:
                            print(f"  {batch:>5} {seqlen:>7} │ skip (attn_mask 约 {mask_gb:.1f} GiB > --max-mask-gb {args.max_mask_gb})")
                        continue
                    try:
                        q = torch.randn(batch, seqlen, nheads_q, headdim, dtype=dtype, device=device)
                        k = torch.randn(batch, seqlen, num_heads_kv, headdim, dtype=dtype, device=device)
                        v = torch.randn(batch, seqlen, num_heads_kv, headdim, dtype=dtype, device=device)
                        attn_mask = torch.ones(batch, nheads_q, seqlen, seqlen, dtype=torch.bool, device=device)
                    except torch.cuda.OutOfMemoryError:
                        if args.table:
                            oom_row = f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\tOOM\t-\tOOM\t-\t-"
                            print(oom_row, flush=True)
                            if run_bwd:
                                rows_bwd.append(oom_row)
                        else:
                            print(f"  {batch:>5} {seqlen:>7} │ OOM (attn_mask 约 {mask_gb:.1f} GiB)")
                        torch.cuda.empty_cache()
                        continue

                    try:
                        t_no = _time_forward_ms(flash_attn_func, q, k, v, causal=causal, repeats=repeats)
                        t_mask = _time_forward_ms(flash_attn_with_mask_func, q, k, v, attn_mask, causal=causal, repeats=repeats)
                    except torch.cuda.OutOfMemoryError:
                        if args.table:
                            oom_row = f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\tOOM\t-\tOOM\t-\t-"
                            print(oom_row, flush=True)
                            if run_bwd:
                                rows_bwd.append(oom_row)
                        else:
                            print(f"  {batch:>5} {seqlen:>7} │ OOM (forward)")
                        del q, k, v, attn_mask
                        torch.cuda.empty_cache()
                        continue

                    ratio_fwd = t_mask / t_no if t_no > 0 else 0.0

                    if args.table:
                        flop_fwd = flops(batch, seqlen, headdim, nheads_q, causal, mode="fwd")
                        tflops_no_fwd = efficiency(flop_fwd, t_no / 1000.0)
                        tflops_attnmask_fwd = efficiency(flop_fwd, t_mask / 1000.0)
                        fwd_pct = (tflops_attnmask_fwd / tflops_no_fwd * 100.0) if tflops_no_fwd > 0 else 0.0
                        if run_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                            try:
                                (no_fwd, no_bwd) = _time_fwd_bwd_ms(flash_attn_func, q, k, v, causal=causal, repeats=repeats)
                                q2 = q.detach().clone().requires_grad_(True)
                                k2 = k.detach().clone().requires_grad_(True)
                                v2 = v.detach().clone().requires_grad_(True)
                                (mask_fwd, mask_bwd) = _time_fwd_bwd_ms(
                                    flash_attn_with_mask_func, q2, k2, v2, attn_mask, causal=causal, repeats=repeats
                                )
                            except torch.cuda.OutOfMemoryError:
                                print(f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\t{tflops_attnmask_fwd:.2f}\t{t_mask:.2f}\t{tflops_no_fwd:.2f}\t{t_no:.2f}\t{fwd_pct:.1f}%", flush=True)
                                rows_bwd.append(f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\tOOM\t-\tOOM\t-\t-")
                                torch.cuda.empty_cache()
                                continue
                            flop_bwd = flops(batch, seqlen, headdim, nheads_q, causal, mode="bwd")
                            tflops_no_bwd = efficiency(flop_bwd, no_bwd / 1000.0)
                            tflops_attnmask_bwd = efficiency(flop_bwd, mask_bwd / 1000.0)
                            bwd_pct = (tflops_attnmask_bwd / tflops_no_bwd * 100.0) if tflops_no_bwd > 0 else 0.0
                            print(f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\t{tflops_attnmask_fwd:.2f}\t{mask_fwd:.2f}\t{tflops_no_fwd:.2f}\t{no_fwd:.2f}\t{fwd_pct:.1f}%", flush=True)
                            rows_bwd.append(f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\t{tflops_attnmask_bwd:.2f}\t{mask_bwd:.2f}\t{tflops_no_bwd:.2f}\t{no_bwd:.2f}\t{bwd_pct:.1f}%")
                        else:
                            print(f"{batch}\t{seqlen}\t{seqlen}\t{nheads_q}\t{num_heads_kv}\t{causal}\t{headdim}\t{dtype_str}\t{tflops_attnmask_fwd:.2f}\t{t_mask:.2f}\t{tflops_no_fwd:.2f}\t{t_no:.2f}\t{fwd_pct:.1f}%", flush=True)
                        continue

                    if run_bwd:
                        q.requires_grad_(True)
                        k.requires_grad_(True)
                        v.requires_grad_(True)
                        (no_fwd, no_bwd) = _time_fwd_bwd_ms(flash_attn_func, q, k, v, causal=causal, repeats=repeats)
                        q2 = q.detach().clone().requires_grad_(True)
                        k2 = k.detach().clone().requires_grad_(True)
                        v2 = v.detach().clone().requires_grad_(True)
                        (mask_fwd, mask_bwd) = _time_fwd_bwd_ms(
                            flash_attn_with_mask_func, q2, k2, v2, attn_mask, causal=causal, repeats=repeats
                        )
                        ratio_fwd = mask_fwd / no_fwd if no_fwd > 0 else 0.0
                        ratio_bwd = mask_bwd / no_bwd if no_bwd > 0 else 0.0
                        print(f"  {batch:>5} {seqlen:>7} │ {no_fwd:>12.3f} {mask_fwd:>12.3f} {ratio_fwd:>8.2f}x │ "
                              f"{no_bwd:>12.3f} {mask_bwd:>12.3f} {ratio_bwd:>8.2f}x")
                    else:
                        print(f"  {batch:>5} {seqlen:>7} │ {t_no:>14.3f} {t_mask:>14.3f} │ {ratio_fwd:>9.2f}x")

            if args.table and run_bwd and rows_bwd:
                print(bwd_header, flush=True)
                for r in rows_bwd:
                    print(r, flush=True)

            if not args.table:
                print("=" * 90)
                print("speed_ratio = attnmask_time / no_attnmask_time  (>1 表示 attnmask 更慢)")
                print()

if __name__ == "__main__":
    main()
