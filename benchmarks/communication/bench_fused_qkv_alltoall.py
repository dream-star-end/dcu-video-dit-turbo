from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass


rank = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = str(rank)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("ROCR_VISIBLE_DEVICES", None)

import torch
import torch.distributed as dist


torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
RANK = dist.get_rank()
WORLD = dist.get_world_size()
assert WORLD == 2
L, H, D = 11819, 56, 128
HP = H // WORLD
S = L * WORLD


def current_one(tensor: torch.Tensor) -> torch.Tensor:
    send = tensor.reshape(L, WORLD, HP, D).permute(1, 0, 2, 3).contiguous().reshape(-1)
    receive = torch.empty(S * HP * D, device=tensor.device, dtype=tensor.dtype)
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[L * HP * D] * WORLD,
        input_split_sizes=[L * HP * D] * WORLD,
    )
    return receive.view(S, HP, D)


def current(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return current_one(q), current_one(k), current_one(v)


def fused(qkv: torch.Tensor):
    # [L, 3, WORLD, HP, D] -> destination-major [WORLD, L, 3, HP, D].
    send = qkv.view(L, 3, WORLD, HP, D).permute(2, 0, 1, 3, 4).contiguous().reshape(-1)
    receive = torch.empty(S * 3 * HP * D, device=qkv.device, dtype=qkv.dtype)
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[L * 3 * HP * D] * WORLD,
        input_split_sizes=[L * 3 * HP * D] * WORLD,
    )
    packed = receive.view(S, 3, HP, D)
    return packed[:, 0], packed[:, 1], packed[:, 2]


generator = torch.Generator(device="cuda").manual_seed(20260812 + RANK)
qkv = torch.randn((L, 3, H, D), device="cuda", dtype=torch.bfloat16, generator=generator)
q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

reference = current(q, k, v)
candidate = fused(qkv)
torch.cuda.synchronize()
local_correct = torch.tensor(
    [int(all(torch.equal(a, b) for a, b in zip(reference, candidate)))],
    device="cuda", dtype=torch.int32,
)
dist.all_reduce(local_correct, op=dist.ReduceOp.MIN)
strides = {
    "current": [list(x.stride()) for x in reference],
    "fused": [list(x.stride()) for x in candidate],
}
del reference, candidate


def measure(fn, repeats=6):
    dist.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = (time.perf_counter() - started) * 1000.0 / repeats
    value = torch.tensor([elapsed], device="cuda", dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    del out
    return float(value.item())


for _ in range(6):
    current(q, k, v)
    fused(qkv)
torch.cuda.synchronize()

current_samples, fused_samples = [], []
for index in range(11):
    if index % 2 == 0:
        current_samples.append(measure(lambda: current(q, k, v)))
        fused_samples.append(measure(lambda: fused(qkv)))
    else:
        fused_samples.append(measure(lambda: fused(qkv)))
        current_samples.append(measure(lambda: current(q, k, v)))

if RANK == 0:
    c = statistics.median(current_samples)
    f = statistics.median(fused_samples)
    saved = c - f
    print(json.dumps({
        "correct": bool(local_correct.item()),
        "shape": {"local_sequence": L, "heads": H, "head_dim": D},
        "strides": strides,
        "current_ms": {"samples": current_samples, "median": c},
        "fused_ms": {"samples": fused_samples, "median": f},
        "saved_ms_per_block": saved,
        "predicted_full20_saved_seconds": saved * 50 * 20 / 1000.0,
    }, indent=2, sort_keys=True), flush=True)
dist.destroy_process_group()
