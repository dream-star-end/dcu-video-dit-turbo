#!/usr/bin/env python3
"""Strict BF16 H3 attention sweep using the real non-contiguous SP strides."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import torch


COMFY_ROOT = "/path/to/ComfyUI"
sys.path.insert(0, COMFY_ROOT)

from comfy.ldm.modules.sub_quadratic_attention import efficient_dot_product_attention  # noqa: E402


HEADS = 28
TOKENS = 23569
DIM = 128
CHUNKS = (384, 448, 512, 576, 640, 704)
REPEATS = 4


def synchronize() -> None:
    torch.cuda.synchronize()


def original(query: torch.Tensor, key_t: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return efficient_dot_product_attention(
        query,
        key_t,
        value,
        query_chunk_size=512,
        kv_chunk_size=TOKENS,
        use_checkpoint=False,
        upcast_attention=False,
        mask=None,
    )


def cached_singleton(
    query: torch.Tensor,
    key_t: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int,
    singleton: torch.Tensor,
) -> torch.Tensor:
    scale = DIM ** -0.5
    outputs = []
    for start in range(0, TOKENS, chunk_size):
        stop = min(start + chunk_size, TOKENS)
        scores = torch.baddbmm(singleton, query[:, start:stop], key_t, alpha=scale, beta=0)
        probabilities = scores.softmax(dim=-1)
        outputs.append(torch.bmm(probabilities, value))
    return torch.cat(outputs, dim=1)


def timed(function) -> tuple[torch.Tensor, list[float]]:
    warm = function()
    synchronize()
    del warm
    samples = []
    output = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = function()
        synchronize()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return output, samples


def main() -> None:
    torch.manual_seed(20260812)
    device = torch.device("cuda", 0)
    q_storage = torch.randn((TOKENS, HEADS, DIM), device=device, dtype=torch.bfloat16)
    k_storage = torch.randn_like(q_storage)
    v_storage = torch.randn_like(q_storage)
    query = q_storage.transpose(0, 1)
    key_t = k_storage.transpose(0, 1).movedim(1, 2)
    value = v_storage.transpose(0, 1)
    singleton = torch.empty((1, 1, 1), device=device, dtype=torch.bfloat16)
    synchronize()

    reference, reference_samples = timed(lambda: original(query, key_t, value))
    reference_median = statistics.median(reference_samples)
    rows = [{
        "variant": "original_allocating_singleton",
        "chunk": 512,
        "samples_seconds": reference_samples,
        "median_seconds": reference_median,
        "speedup_percent": 0.0,
        "bitwise_equal": True,
        "max_abs_diff": 0.0,
    }]

    for chunk in CHUNKS:
        output, samples = timed(
            lambda chunk=chunk: cached_singleton(query, key_t, value, chunk, singleton)
        )
        median = statistics.median(samples)
        rows.append({
            "variant": "cached_singleton",
            "chunk": chunk,
            "samples_seconds": samples,
            "median_seconds": median,
            "speedup_percent": (reference_median / median - 1.0) * 100.0,
            "bitwise_equal": torch.equal(reference, output),
            "max_abs_diff": float((reference.float() - output.float()).abs().max().item()),
        })
        del output

    print(json.dumps({
        "candidate": "strict_subquadratic_stride_and_singleton_sweep",
        "shape": {
            "heads": HEADS,
            "tokens": TOKENS,
            "dim": DIM,
            "dtype": "torch.bfloat16",
            "query_stride": list(query.stride()),
            "key_t_stride": list(key_t.stride()),
            "value_stride": list(value.stride()),
        },
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
