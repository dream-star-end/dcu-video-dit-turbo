#!/usr/bin/env python3
"""Compare production-shape H3 sub-quadratic attention with ROCm Flash SDPA."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


COMFY_ROOT = "/path/to/ComfyUI"
sys.path.insert(0, COMFY_ROOT)

from comfy.ldm.modules.sub_quadratic_attention import efficient_dot_product_attention  # noqa: E402


HEADS = 28
TOKENS = 23569
DIM = 128
REPEATS = 4


def synchronize() -> None:
    torch.cuda.synchronize()


def subquadratic(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    query = q.reshape(HEADS, TOKENS, DIM)
    key_t = k.reshape(HEADS, TOKENS, DIM).movedim(1, 2)
    value = v.reshape(HEADS, TOKENS, DIM)
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


def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )[0]


def timed(function, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    warm = function(q, k, v)
    synchronize()
    del warm
    samples = []
    output = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = function(q, k, v)
        synchronize()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return output, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("default", "cutlass", "aotriton", "ck"), required=True)
    args = parser.parse_args()

    torch.backends.cuda.preferred_rocm_fa_library(args.backend)
    torch.manual_seed(20260812)
    device = torch.device("cuda", 0)
    # Match the sequence-major storage and head-major non-contiguous views used
    # by the real sequence-parallel model.
    q_storage = torch.randn((TOKENS, HEADS, DIM), device=device, dtype=torch.bfloat16)
    k_storage = torch.randn_like(q_storage)
    v_storage = torch.randn_like(q_storage)
    q = q_storage.transpose(0, 1).unsqueeze(0)
    k = k_storage.transpose(0, 1).unsqueeze(0)
    v = v_storage.transpose(0, 1).unsqueeze(0)
    synchronize()

    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
    flash_usable = torch.backends.cuda.can_use_flash_attention(params, debug=True)
    if not flash_usable:
        raise RuntimeError("Flash Attention is unavailable for the production H3 shape")

    reference, reference_samples = timed(subquadratic, q, k, v)
    candidate, candidate_samples = timed(flash, q, k, v)
    delta = reference.float() - candidate.float()
    reference_float = reference.float()
    rmse = float(delta.square().mean().sqrt().item())
    reference_rms = float(reference_float.square().mean().sqrt().item())
    metrics = {
        "bitwise_equal": torch.equal(reference, candidate),
        "max_abs_diff": float(delta.abs().max().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms,
        "cosine_similarity": float(F.cosine_similarity(
            reference_float.reshape(1, -1),
            candidate.float().reshape(1, -1),
        ).item()),
    }
    reference_median = statistics.median(reference_samples)
    candidate_median = statistics.median(candidate_samples)
    print(json.dumps({
        "candidate": "rocm_flash_sdpa",
        "backend_requested": args.backend,
        "backend_selected": str(torch.backends.cuda.preferred_rocm_fa_library()),
        "shape": {
            "batch": 1,
            "heads": HEADS,
            "tokens": TOKENS,
            "dim": DIM,
            "dtype": "torch.bfloat16",
            "input_stride": list(q.stride()),
        },
        "flash_usable": flash_usable,
        "reference": {
            "algorithm": "subquadratic_query_chunk_512",
            "samples_seconds": reference_samples,
            "median_seconds": reference_median,
        },
        "candidate_result": {
            "samples_seconds": candidate_samples,
            "median_seconds": candidate_median,
            "speedup_x": reference_median / candidate_median,
            "speedup_percent": (reference_median / candidate_median - 1.0) * 100.0,
        },
        "numeric_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
