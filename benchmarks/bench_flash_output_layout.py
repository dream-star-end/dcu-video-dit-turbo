#!/usr/bin/env python3
"""Measure the H3 output-layout conversion around a FlashAttention module."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time

import torch


HEADS = 28
TOKENS = 23638
DIM = 128


def load(path: str):
    spec = importlib.util.spec_from_file_location("h3_flash_attn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(function, repeats: int = 12):
    outputs = [function() for _ in range(3)]
    torch.cuda.synchronize()
    del outputs
    torch.cuda.reset_peak_memory_stats()
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return output, {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    module = load(args.so)
    torch.manual_seed(20260812)
    storages = [
        torch.randn((TOKENS, HEADS, DIM), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    q, k, v = (tensor.transpose(0, 1).unsqueeze(0) for tensor in storages)

    raw, raw_timing = timed(lambda: module.fwd_bhsd(q, k, v, DIM ** -0.5))
    converted, converted_timing = timed(
        lambda: (
            module.fwd_bhsd(q, k, v, DIM ** -0.5)
            .transpose(1, 2)
            .flatten(start_dim=2)
        )
    )
    raw_as_h3 = raw.transpose(1, 2).flatten(start_dim=2)
    print(json.dumps({
        "label": args.label,
        "shape": [1, HEADS, TOKENS, DIM],
        "raw": {
            **raw_timing,
            "stride": list(raw.stride()),
            "is_contiguous": raw.is_contiguous(),
        },
        "h3_converted": {
            **converted_timing,
            "stride": list(converted.stride()),
            "is_contiguous": converted.is_contiguous(),
            "shares_storage_with_raw_view": raw_as_h3.untyped_storage().data_ptr()
            == raw.untyped_storage().data_ptr(),
        },
        "conversion_overhead_milliseconds": (
            converted_timing["median_seconds"] - raw_timing["median_seconds"]
        ) * 1000.0,
    }, indent=2))


if __name__ == "__main__":
    main()
