#!/usr/bin/env python3
"""Benchmark lossless Q-only chunking and output workspace reuse for H3 SP."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import torch


COMFY_ROOT = "/path/to/ComfyUI"
sys.path.insert(0, COMFY_ROOT)

from comfy.ldm.modules.sub_quadratic_attention import (  # noqa: E402
    _get_attention_scores_no_kv_chunking,
    efficient_dot_product_attention,
)


HEADS = 28
TOKENS = 23569
DIM = 128
CHUNKS = (512, 768, 1024, 1536)
REPEATS = 3


def sync() -> None:
    torch.cuda.synchronize()


def original(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, chunk: int) -> torch.Tensor:
    return efficient_dot_product_attention(
        q,
        kt,
        v,
        query_chunk_size=chunk,
        kv_chunk_size=TOKENS,
        use_checkpoint=False,
        upcast_attention=False,
        mask=None,
    )


def preallocated(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, chunk: int) -> torch.Tensor:
    out = torch.empty_like(q)
    scale = DIM ** -0.5
    for start in range(0, TOKENS, chunk):
        stop = min(start + chunk, TOKENS)
        out[:, start:stop].copy_(
            _get_attention_scores_no_kv_chunking(
                query=q[:, start:stop],
                key_t=kt,
                value=v,
                scale=scale,
                upcast_attention=False,
                mask=None,
            )
        )
    return out


def timed(fn, q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, chunk: int):
    warm = fn(q, kt, v, chunk)
    sync()
    del warm
    samples = []
    peak = 0
    last = None
    for _ in range(REPEATS):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        last = fn(q, kt, v, chunk)
        sync()
        samples.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated())
    return last, samples, peak


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HIP device is not available")
    torch.manual_seed(20260812)
    device = torch.device("cuda", 0)
    q = torch.randn((HEADS, TOKENS, DIM), device=device, dtype=torch.bfloat16)
    k = torch.randn((HEADS, TOKENS, DIM), device=device, dtype=torch.bfloat16)
    v = torch.randn((HEADS, TOKENS, DIM), device=device, dtype=torch.bfloat16)
    kt = k.transpose(1, 2)
    sync()

    reference, ref_times, ref_peak = timed(original, q, kt, v, 512)
    rows = [{
        "variant": "original",
        "chunk": 512,
        "median_seconds": statistics.median(ref_times),
        "samples_seconds": ref_times,
        "peak_allocated_bytes": ref_peak,
        "bitwise_equal_to_reference": True,
        "max_abs_diff": 0.0,
    }]

    for variant, fn in (("original", original), ("preallocated", preallocated)):
        for chunk in CHUNKS:
            if variant == "original" and chunk == 512:
                continue
            output, samples, peak = timed(fn, q, kt, v, chunk)
            equal = torch.equal(reference, output)
            max_abs = float((reference.float() - output.float()).abs().max().item())
            rows.append({
                "variant": variant,
                "chunk": chunk,
                "median_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "peak_allocated_bytes": peak,
                "bitwise_equal_to_reference": equal,
                "max_abs_diff": max_abs,
            })
            del output

    print(json.dumps({
        "shape": {"heads": HEADS, "tokens": TOKENS, "dim": DIM},
        "dtype": str(q.dtype),
        "upcast_attention": False,
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
