from __future__ import annotations

import json
import os
import statistics
import time

rank = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = str(rank)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("ROCR_VISIBLE_DEVICES", None)

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
RANK, WORLD = dist.get_rank(), dist.get_world_size()
L, HP, D = 11819, 28, 128
S = L * WORLD


def collective_raw(tensor):
    send = tensor.reshape(-1)
    receive = torch.empty(WORLD * L * HP * D, device="cuda", dtype=tensor.dtype)
    dist.all_to_all_single(
        receive,
        send,
        output_split_sizes=[L * HP * D] * WORLD,
        input_split_sizes=[L * HP * D] * WORLD,
    )
    return receive.view(WORLD, L, HP, D)


def current(tensor):
    receive = collective_raw(tensor)
    return receive.permute(1, 0, 2, 3).reshape(L, WORLD * HP, D)


g = torch.Generator(device="cuda").manual_seed(20260812 + RANK)
x = torch.randn((S, HP, D), device="cuda", dtype=torch.bfloat16, generator=g)
ref = current(x)
raw = collective_raw(x)
reconstructed = raw.permute(1, 0, 2, 3).reshape(L, WORLD * HP, D)
torch.cuda.synchronize()
ok = torch.tensor([int(torch.equal(ref, reconstructed))], device="cuda", dtype=torch.int32)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
del ref, raw, reconstructed


def measure(fn, repeats=8):
    dist.barrier(); torch.cuda.synchronize(); start = time.perf_counter()
    out = None
    for _ in range(repeats): out = fn(x)
    torch.cuda.synchronize(); dist.barrier()
    ms = (time.perf_counter() - start) * 1000.0 / repeats
    value = torch.tensor([ms], device="cuda", dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    del out
    return float(value.item())


for _ in range(6): current(x); collective_raw(x)
torch.cuda.synchronize()
current_samples, raw_samples = [], []
for i in range(11):
    if i % 2 == 0:
        current_samples.append(measure(current)); raw_samples.append(measure(collective_raw))
    else:
        raw_samples.append(measure(collective_raw)); current_samples.append(measure(current))

if RANK == 0:
    c, r = statistics.median(current_samples), statistics.median(raw_samples)
    delta = c - r
    print(json.dumps({
        "correct_reconstruction": bool(ok.item()),
        "current_ms": {"samples": current_samples, "median": c},
        "raw_collective_ms": {"samples": raw_samples, "median": r},
        "unpack_copy_ceiling_ms_per_block": delta,
        "full20_absolute_ceiling_seconds": delta * 50 * 20 / 1000.0,
    }, indent=2, sort_keys=True), flush=True)
dist.destroy_process_group()
