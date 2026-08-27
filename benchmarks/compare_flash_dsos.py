#!/usr/bin/env python3
"""Bitwise and timing comparison for two H3 FlashAttention DSOs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time

import torch


def load_extension(path: str):
    spec = importlib.util.spec_from_file_location("h3_flash_attn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def timed(callable_, repeats: int) -> tuple[torch.Tensor, dict]:
    output = None
    for _ in range(3):
        output = callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = callable_()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return output, {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    reference_module = load_extension(args.reference)
    candidate_module = load_extension(args.candidate)
    results = []
    for tokens in args.tokens:
        torch.manual_seed(20260812)
        storages = [
            torch.randn(
                (tokens, args.heads, 128),
                device="cuda:0",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        ]
        q, k, v = (value.transpose(0, 1).unsqueeze(0) for value in storages)
        scale = 128 ** -0.5

        reference_output, reference_timing = timed(
            lambda: reference_module.fwd_bhsd(q, k, v, scale), args.repeats
        )
        candidate_output, candidate_timing = timed(
            lambda: candidate_module.fwd_bhsd(q, k, v, scale), args.repeats
        )
        delta = reference_output.float() - candidate_output.float()
        results.append({
            "shape": list(q.shape),
            "input_stride": list(q.stride()),
            "reference_output_stride": list(reference_output.stride()),
            "candidate_output_stride": list(candidate_output.stride()),
            "bitwise_equal": bool(torch.equal(reference_output, candidate_output)),
            "reference_sha256": tensor_sha256(reference_output),
            "candidate_sha256": tensor_sha256(candidate_output),
            "max_abs_diff": float(delta.abs().max().item()),
            "mean_abs_diff": float(delta.abs().mean().item()),
            "reference_timing": reference_timing,
            "candidate_timing": candidate_timing,
            "speedup": (
                reference_timing["median_seconds"]
                / candidate_timing["median_seconds"]
            ),
        })
        del storages, q, k, v, reference_output, candidate_output, delta
        torch.cuda.empty_cache()

    payload = {
        "reference": args.reference,
        "candidate": args.candidate,
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
