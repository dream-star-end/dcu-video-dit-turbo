#!/usr/bin/env python3
"""Hot two-rank A/B benchmark for packed H3 QKV all-to-all."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist


SEQUENCE_SPLITS = (11785, 11784)
HEADS = 56
DIM = 128
ROUNDS = 4


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("h3_sequence_parallel_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sequence-parallel candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synchronize() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def run_reference(context, qkv: torch.Tensor):
    q, k, v = qkv.unbind(dim=1)
    return tuple(context.sequence_to_heads(tensor) for tensor in (q, k, v))


def run_candidate(context, qkv: torch.Tensor):
    return context.sequence_to_heads_qkv(qkv)


def timed(label: str, function, context, qkv: torch.Tensor) -> tuple[str, float]:
    synchronize()
    start = time.perf_counter()
    output = function(context, qkv)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del output
    return label, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    module = load_module(args.module)
    rank = dist.get_rank()
    context = module.AttentionSequenceParallel(
        rank=rank,
        world_size=2,
        sequence_splits=SEQUENCE_SPLITS,
    )
    qkv = torch.empty(
        (SEQUENCE_SPLITS[rank], 3, HEADS, DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Correctness at the production tensor shape before timing.
    reference = run_reference(context, qkv)
    candidate = run_candidate(context, qkv)
    bitwise = [torch.equal(a, b) for a, b in zip(reference, candidate)]
    stride_equal = [a.stride() == b.stride() for a, b in zip(reference, candidate)]
    del reference, candidate
    if not all(bitwise) or not all(stride_equal):
        raise RuntimeError(f"production-shape contract failed: bitwise={bitwise}, stride={stride_equal}")

    # Warm both arms, then alternate order to reduce clock/temperature bias.
    for function in (run_reference, run_candidate):
        output = function(context, qkv)
        torch.cuda.synchronize()
        del output
    schedule = (
        (("reference", run_reference), ("candidate", run_candidate)),
        (("candidate", run_candidate), ("reference", run_reference)),
    )
    samples = {"reference": [], "candidate": []}
    for round_index in range(ROUNDS):
        for label, function in schedule[round_index % len(schedule)]:
            measured_label, elapsed = timed(label, function, context, qkv)
            samples[measured_label].append(elapsed)

    local_payload = {
        "rank": rank,
        "bitwise_equal_qkv": bitwise,
        "stride_equal_qkv": stride_equal,
        "samples_seconds": samples,
        "median_seconds": {
            label: statistics.median(values) for label, values in samples.items()
        },
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_payload)
    if rank == 0:
        worst_reference = max(item["median_seconds"]["reference"] for item in gathered)
        worst_candidate = max(item["median_seconds"]["candidate"] for item in gathered)
        print(json.dumps({
            "candidate": "packed_qkv_all_to_all",
            "shape": {
                "sequence_splits": list(SEQUENCE_SPLITS),
                "heads": HEADS,
                "dim": DIM,
                "dtype": "torch.bfloat16"
            },
            "rounds_per_arm": ROUNDS,
            "ranks": gathered,
            "worst_rank_median_seconds": {
                "reference": worst_reference,
                "candidate": worst_candidate
            },
            "speedup_percent": (worst_reference / worst_candidate - 1.0) * 100.0
        }, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
