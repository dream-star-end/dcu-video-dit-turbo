#!/usr/bin/env python3
"""Two-rank real-shape benchmark for fused-QKV head-chunk SP overlap.

This is a campaign-only probe.  It does not import or modify the live H3
runtime.  Each torchrun rank exposes one physical accelerator, reproduces the
current Ulysses byte layout, and calls the audited H3 FlashAttention DSO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import time
from dataclasses import dataclass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-so", required=True)
    parser.add_argument("--local-length", type=int, default=11819)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


ARGS = parse_args()
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
os.environ["HIP_VISIBLE_DEVICES"] = str(LOCAL_RANK)
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def load_flash(path: str):
    spec = importlib.util.spec_from_file_location("h3_flash_attn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FlashAttention DSO: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PendingChunk:
    receive: torch.Tensor
    send: torch.Tensor
    work: dist.Work


class Probe:
    def __init__(self, flash, local_length: int, heads: int, dim: int):
        self.flash = flash
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        if self.world != 2:
            raise RuntimeError(f"probe requires two ranks, got {self.world}")
        if heads % self.world:
            raise RuntimeError("heads must divide world size")
        self.splits = (local_length, local_length)
        self.local_length = local_length
        self.sequence = sum(self.splits)
        self.heads = heads
        self.local_heads = heads // self.world
        self.dim = dim
        self.scale = dim ** -0.5
        self.comm_stream = torch.cuda.Stream(device="cuda")

        # Use bounded integer values so BF16 conversion is exact and equality
        # failures diagnose ordering/stream hazards rather than input rounding.
        generator = torch.Generator(device="cuda").manual_seed(20260812 + self.rank)
        shape = (local_length, heads, dim)
        self.q = torch.randint(-8, 9, shape, generator=generator, device="cuda", dtype=torch.int16).to(torch.bfloat16)
        self.k = torch.randint(-8, 9, shape, generator=generator, device="cuda", dtype=torch.int16).to(torch.bfloat16)
        self.v = torch.randint(-8, 9, shape, generator=generator, device="cuda", dtype=torch.int16).to(torch.bfloat16)

    def sequence_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        send = tensor.reshape(
            self.local_length, self.world, self.local_heads, self.dim
        ).permute(1, 0, 2, 3).contiguous().reshape(-1)
        receive = torch.empty(
            self.sequence * self.local_heads * self.dim,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        dist.all_to_all_single(
            receive,
            send,
            output_split_sizes=[size * self.local_heads * self.dim for size in self.splits],
            input_split_sizes=[self.local_length * self.local_heads * self.dim] * self.world,
        )
        return receive.view(self.sequence, self.local_heads, self.dim)

    def heads_to_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        chunks = tensor.split(self.splits, dim=0)
        send = torch.cat([chunk.contiguous().reshape(-1) for chunk in chunks])
        receive = torch.empty(
            self.world * self.local_length * self.local_heads * self.dim,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        dist.all_to_all_single(
            receive,
            send,
            output_split_sizes=[self.local_length * self.local_heads * self.dim] * self.world,
            input_split_sizes=[size * self.local_heads * self.dim for size in self.splits],
        )
        return receive.view(
            self.world, self.local_length, self.local_heads, self.dim
        ).permute(1, 0, 2, 3).reshape(
            self.local_length, self.heads, self.dim
        )

    def flash_model_layout(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        head_count = q.shape[1]
        out = self.flash.fwd_bhsd(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            self.scale,
        )
        # Reproduce the launcher's output path exactly.  Depending on the DSO,
        # this is either a view or the same layout copy paid by the live worker.
        return out.transpose(1, 2).flatten(start_dim=2).squeeze(0).view(
            self.sequence, head_count, self.dim
        )

    def baseline(self) -> torch.Tensor:
        qh = self.sequence_to_heads(self.q)
        kh = self.sequence_to_heads(self.k)
        vh = self.sequence_to_heads(self.v)
        return self.heads_to_sequence(self.flash_model_layout(qh, kh, vh))

    def _pack_chunk(self, start: int, stop: int, async_op: bool) -> PendingChunk:
        count = stop - start
        send = torch.empty(
            (self.world, self.local_length, count, 3, self.dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        for channel, tensor in enumerate((self.q, self.k, self.v)):
            source = tensor.reshape(
                self.local_length, self.world, self.local_heads, self.dim
            ).permute(1, 0, 2, 3)
            send[:, :, :, channel, :].copy_(source[:, :, start:stop, :])
        receive = torch.empty(
            (self.sequence, count, 3, self.dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        work = dist.all_to_all_single(
            receive.reshape(-1),
            send.reshape(-1),
            output_split_sizes=[size * count * 3 * self.dim for size in self.splits],
            input_split_sizes=[self.local_length * count * 3 * self.dim] * self.world,
            async_op=async_op,
        )
        return PendingChunk(receive, send, work)

    def fused_one(self) -> torch.Tensor:
        pending = self._pack_chunk(0, self.local_heads, async_op=False)
        qh, kh, vh = pending.receive.unbind(dim=2)
        return self.heads_to_sequence(self.flash_model_layout(qh, kh, vh))

    def fused_two_serial(self) -> torch.Tensor:
        outputs = []
        half = self.local_heads // 2
        for start, stop in ((0, half), (half, self.local_heads)):
            pending = self._pack_chunk(start, stop, async_op=False)
            outputs.append(self.flash_model_layout(*pending.receive.unbind(dim=2)))
        return self.heads_to_sequence(torch.cat(outputs, dim=1))

    def fused_two_overlap(self) -> torch.Tensor:
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        half = self.local_heads // 2
        pending = []
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(ready)
            pending.append(self._pack_chunk(0, half, async_op=True))
            pending.append(self._pack_chunk(half, self.local_heads, async_op=True))

        outputs = []
        for item in pending:
            item.work.block_current_stream()
            outputs.append(self.flash_model_layout(*item.receive.unbind(dim=2)))
        return self.heads_to_sequence(torch.cat(outputs, dim=1))


def all_rank_bool(value: bool) -> bool:
    flag = torch.tensor([int(value)], device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def all_rank_max(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    equal = torch.equal(reference, candidate)
    max_diff = 0.0
    if not equal:
        for part_ref, part_candidate in zip(reference.split(1024), candidate.split(1024)):
            max_diff = max(
                max_diff,
                float((part_ref.float() - part_candidate.float()).abs().max().item()),
            )
    return {
        "bitwise_equal_all_ranks": all_rank_bool(equal),
        "max_abs_diff_all_ranks": all_rank_max(max_diff),
    }


def benchmark(name: str, function, warmup: int, repeats: int) -> dict:
    samples = []
    for iteration in range(warmup + repeats):
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        result = function()
        # Force the reverse A2A/output and all async dependencies to finish.
        del result
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        elapsed = all_rank_max(elapsed)
        if iteration >= warmup:
            samples.append(elapsed * 1000.0)
    return {
        "name": name,
        "samples_ms_worst_rank": samples,
        "median_ms_worst_rank": statistics.median(samples),
        "min_ms_worst_rank": min(samples),
        "max_ms_worst_rank": max(samples),
    }


def benchmark_flash(name: str, flash, q, k, v, scale, warmup: int, repeats: int) -> dict:
    def call():
        return flash.fwd_bhsd(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            scale,
        )

    return benchmark(name, call, warmup, repeats)


def main() -> None:
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    flash = load_flash(ARGS.flash_so)
    probe = Probe(flash, ARGS.local_length, ARGS.heads, ARGS.dim)

    # Correctness first: compare the complete reverse-transposed attention output.
    baseline_output = probe.baseline()
    fused_one_output = probe.fused_one()
    fused_serial_output = probe.fused_two_serial()
    fused_overlap_output = probe.fused_two_overlap()
    correctness = {
        "fused_one": compare(baseline_output, fused_one_output),
        "fused_two_serial": compare(baseline_output, fused_serial_output),
        "fused_two_overlap": compare(baseline_output, fused_overlap_output),
    }

    # Isolate the only new Flash contract: H=14 and QKV interleaved at stride 3D.
    half = probe.local_heads // 2
    packed = probe._pack_chunk(0, half, async_op=False).receive
    interleaved = packed.unbind(dim=2)
    standard = tuple(tensor.contiguous() for tensor in interleaved)
    standard_out = probe.flash_model_layout(*standard)
    interleaved_out = probe.flash_model_layout(*interleaved)
    flash_stride_correctness = compare(standard_out, interleaved_out)

    timings = [
        benchmark("baseline_3a2a_flash28_reverse_a2a", probe.baseline, ARGS.warmup, ARGS.repeats),
        benchmark("fused1_1a2a_flash28_reverse_a2a", probe.fused_one, ARGS.warmup, ARGS.repeats),
        benchmark("fused2_serial_2a2a_2xflash14_reverse_a2a", probe.fused_two_serial, ARGS.warmup, ARGS.repeats),
        benchmark("fused2_overlap_2a2a_2xflash14_reverse_a2a", probe.fused_two_overlap, ARGS.warmup, ARGS.repeats),
        benchmark_flash("flash14_standard_stride", flash, *standard, probe.scale, ARGS.warmup, ARGS.repeats),
        benchmark_flash("flash14_interleaved_stride", flash, *interleaved, probe.scale, ARGS.warmup, ARGS.repeats),
    ]

    payload = {
        "candidate": "fused_qkv_two_head_chunks_async_overlap",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "world_size": dist.get_world_size(),
        "shape_per_rank": {
            "local_sequence": probe.local_length,
            "global_sequence": probe.sequence,
            "global_heads": probe.heads,
            "heads_per_rank": probe.local_heads,
            "chunk_heads": half,
            "head_dim": probe.dim,
            "dtype": "torch.bfloat16",
        },
        "correctness": correctness,
        "flash14_interleaved_correctness": flash_stride_correctness,
        "timings": timings,
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() == 0:
        # Timing entries already use the cross-rank maximum; emit one canonical payload.
        print(json.dumps(payload, indent=2))

    passed = all(
        result["bitwise_equal_all_ranks"]
        for result in correctness.values()
    ) and flash_stride_correctness["bitwise_equal_all_ranks"]
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
