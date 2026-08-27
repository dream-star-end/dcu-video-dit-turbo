#!/usr/bin/env python3
"""Numerical and performance gates for the gfx936 OpenDAS H3 kernel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


COMFY_ROOT = "/path/to/ComfyUI"
DEFAULT_SO = (
    "kernels/flash-attention-gfx936/build/"
    "flash_attn_hg_forward_gfx936/libflash_attention.so"
)
DIM = 128
PRODUCTION_HEADS = 28
PRODUCTION_TOKENS = 23638

sys.path.insert(0, COMFY_ROOT)
from comfy.ldm.modules.sub_quadratic_attention import (  # noqa: E402
    efficient_dot_product_attention,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def load_extension(path: str):
    spec = importlib.util.spec_from_file_location("h3_flash_attn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_qkv(batch: int, heads: int, tokens: int, *, strided: bool):
    device = torch.device("cuda", 0)
    if strided:
        if batch != 1:
            raise ValueError("The real H3 SP stride reproducer requires batch=1")
        storages = [
            torch.randn(
                (tokens, heads, DIM),
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        ]
        # Exact H3 SP layout: contiguous [S,H,D], then transpose and add B=1.
        return tuple(t.transpose(0, 1).unsqueeze(0) for t in storages)
    return tuple(
        torch.randn(
            (batch, heads, tokens, DIM),
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )


def exact_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (DIM ** -0.5)
    probabilities = scores.softmax(dim=-1)
    return torch.matmul(probabilities, v.float()).to(torch.bfloat16)


def subquadratic(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, dim = q.shape
    if batch != 1 or dim != DIM:
        raise ValueError("This gate only implements the H3 B=1, D=128 path")
    query = q.reshape(heads, tokens, dim)
    key_t = k.reshape(heads, tokens, dim).movedim(1, 2)
    value = v.reshape(heads, tokens, dim)
    output = efficient_dot_product_attention(
        query,
        key_t,
        value,
        query_chunk_size=512,
        kv_chunk_size=tokens,
        use_checkpoint=False,
        upcast_attention=False,
        mask=None,
    )
    return output.unsqueeze(0)


def numeric_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_float = reference.float()
    candidate_float = candidate.float()
    delta = reference_float - candidate_float
    rmse = delta.square().mean().sqrt()
    reference_rms = reference_float.square().mean().sqrt()
    dot = (reference_float * candidate_float).sum(dtype=torch.float64)
    reference_norm = reference_float.square().sum(dtype=torch.float64).sqrt()
    candidate_norm = candidate_float.square().sum(dtype=torch.float64).sqrt()
    cosine = dot / (reference_norm * candidate_norm)
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "finite_fraction": float(torch.isfinite(candidate_float).float().mean().item()),
        "max_abs_diff": float(delta.abs().max().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_rmse": float((rmse / reference_rms).item()),
        "cosine_similarity": float(cosine.item()),
    }


def timed(function, repeats: int, warmups: int = 2):
    output = None
    for _ in range(warmups):
        output = function()
    synchronize()
    del output
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = function()
        synchronize()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return output, {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def small_gate(module) -> dict:
    rows = []
    for tokens, strided in ((128, False), (129, True), (257, True), (512, True), (1024, True)):
        q, k, v = make_qkv(1, 4, tokens, strided=strided)
        synchronize()
        reference = exact_fp32(q, k, v)
        baseline = subquadratic(q, k, v)
        candidate, timing = timed(
            lambda: module.fwd_bhsd(q, k, v, DIM ** -0.5),
            repeats=8,
            warmups=2,
        )
        rows.append({
            "shape": [1, 4, tokens, DIM],
            "strided": strided,
            "input_stride": list(q.stride()),
            "candidate_timing": timing,
            "candidate_vs_fp32_reference": numeric_metrics(reference, candidate),
            "subquadratic_vs_fp32_reference": numeric_metrics(reference, baseline),
            "candidate_vs_subquadratic": numeric_metrics(baseline, candidate),
        })
        del q, k, v, reference, baseline, candidate
        torch.cuda.empty_cache()
    return {"mode": "small", "results": rows}


def production_gate(module) -> dict:
    q, k, v = make_qkv(
        1,
        PRODUCTION_HEADS,
        PRODUCTION_TOKENS,
        strided=True,
    )
    synchronize()
    baseline, baseline_timing = timed(
        lambda: subquadratic(q, k, v),
        repeats=4,
        warmups=2,
    )
    candidate, candidate_timing = timed(
        lambda: module.fwd_bhsd(q, k, v, DIM ** -0.5),
        repeats=8,
        warmups=3,
    )
    baseline_median = baseline_timing["median_seconds"]
    candidate_median = candidate_timing["median_seconds"]
    return {
        "mode": "production",
        "shape": [1, PRODUCTION_HEADS, PRODUCTION_TOKENS, DIM],
        "input_stride": list(q.stride()),
        "subquadratic": baseline_timing,
        "candidate": {
            **candidate_timing,
            "speedup_x": baseline_median / candidate_median,
            "speedup_percent": (baseline_median / candidate_median - 1.0) * 100.0,
        },
        "candidate_vs_subquadratic": numeric_metrics(baseline, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("small", "production"), required=True)
    parser.add_argument("--so", default=DEFAULT_SO)
    args = parser.parse_args()

    if not Path(args.so).is_file():
        raise FileNotFoundError(args.so)
    torch.manual_seed(20260812)
    module = load_extension(args.so)
    metadata = {
        "candidate": "opendas_hg_bf16_flashattention_gfx936",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "softmax_scale": DIM ** -0.5,
        "dtype": "torch.bfloat16",
        "result": small_gate(module) if args.mode == "small" else production_gate(module),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
