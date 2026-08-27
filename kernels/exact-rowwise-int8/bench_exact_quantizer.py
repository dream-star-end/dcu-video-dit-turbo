from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise as eager_quantize


ROOT = Path(__file__).resolve().parent
SO = ROOT / "build" / "h3_exact_rowwise_int8.so"
spec = importlib.util.spec_from_file_location("h3_exact_rowwise_int8", SO)
extension = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extension)

M = 11819
KS = (5376, 7168, 14336)
SEED = 20260812
WARMUP = 12
ROUNDS = 11
REPEATS = 20


def candidate(x: torch.Tensor):
    return extension.quantize_int8_rowwise(x)


def make_case(kind: str, k: int) -> torch.Tensor:
    if kind == "random":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED + k)
        # Row-varying amplitudes exercise many distinct BF16 maxima/scales.
        x = torch.randn((M, k), device="cuda", dtype=torch.bfloat16, generator=generator)
        amplitudes = torch.linspace(0.03125, 8.0, M, device="cuda", dtype=torch.float32)
        return (x * amplitudes[:, None].to(torch.bfloat16)).contiguous()
    if kind == "all_zero":
        return torch.zeros((M, k), device="cuda", dtype=torch.bfloat16)
    if kind == "extreme_finite":
        finfo = torch.finfo(torch.bfloat16)
        pattern = torch.tensor(
            [
                finfo.max,
                -finfo.max,
                finfo.tiny,
                -finfo.tiny,
                1.0e-30,
                -1.0e-30,
                0.0,
                -0.0,
                1.0,
                -1.0,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
    elif kind == "round_even_boundaries":
        # max(abs(x))=127 gives scale exactly 1.0. Half-integers therefore
        # directly exercise round-to-nearest-even on both signs.
        pattern = torch.tensor(
            [
                127.0,
                -127.0,
                126.5,
                125.5,
                2.5,
                1.5,
                0.5,
                -0.5,
                -1.5,
                -2.5,
                -125.5,
                -126.5,
                0.0,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
    else:
        raise ValueError(kind)
    repeats = (M * k + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(repeats)[: M * k].reshape(M, k).contiguous()


def validate(x: torch.Tensor, kind: str, k: int) -> dict:
    q_ref, s_ref = eager_quantize(x)
    q_new, s_new = candidate(x)
    torch.cuda.synchronize()
    q_mismatch = int(torch.count_nonzero(q_ref != q_new).item())
    scale_mismatch = int(
        torch.count_nonzero(s_ref.view(torch.int32) != s_new.view(torch.int32)).item()
    )
    result = {
        "case": kind,
        "shape": [M, k],
        "q_bitwise_equal": q_mismatch == 0,
        "scale_bitwise_equal": scale_mismatch == 0,
        "q_mismatch_count": q_mismatch,
        "scale_mismatch_count": scale_mismatch,
        "scale_ref_min": float(torch.nan_to_num(s_ref, nan=0.0).min().item()),
        "scale_ref_max": float(torch.nan_to_num(s_ref, nan=0.0).max().item()),
    }
    del q_ref, s_ref, q_new, s_new
    return result


def timed_batch(fn, x: torch.Tensor, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(x)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def benchmark(x: torch.Tensor, k: int) -> dict:
    for _ in range(WARMUP):
        eager_quantize(x)
        candidate(x)
    torch.cuda.synchronize()

    eager_samples = []
    candidate_samples = []
    for round_index in range(ROUNDS):
        if round_index % 2 == 0:
            eager_samples.append(timed_batch(eager_quantize, x, REPEATS))
            candidate_samples.append(timed_batch(candidate, x, REPEATS))
        else:
            candidate_samples.append(timed_batch(candidate, x, REPEATS))
            eager_samples.append(timed_batch(eager_quantize, x, REPEATS))
    eager_median = statistics.median(eager_samples)
    candidate_median = statistics.median(candidate_samples)
    return {
        "shape": [M, k],
        "warmup": WARMUP,
        "rounds": ROUNDS,
        "repeats_per_round": REPEATS,
        "eager_ms": {
            "samples": eager_samples,
            "median": eager_median,
            "min": min(eager_samples),
            "max": max(eager_samples),
        },
        "candidate_ms": {
            "samples": candidate_samples,
            "median": candidate_median,
            "min": min(candidate_samples),
            "max": max(candidate_samples),
        },
        "speedup": eager_median / candidate_median,
        "saved_ms_per_call": eager_median - candidate_median,
    }


def main() -> None:
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    report = {
        "contract": {
            "baseline": "comfy_kitchen.backends.eager.quantization.quantize_int8_rowwise",
            "acceptance": "q and FP32 scale bitwise equal for every case; predicted full20 saving > 1 second",
            "production_calls_per_full20": {"K5376": 2000, "K7168": 1000, "K14336": 1000},
        },
        "environment": {
            "timestamp_unix": time.time(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": props.name,
            "arch": props.gcnArchName,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "compile_flags": [
                "-O3",
                "-fno-fast-math",
                "-fno-unsafe-math-optimizations",
                "-ffp-contract=off",
                "--offload-arch=gfx936",
            ],
        },
        "correctness": [],
        "timing": [],
    }

    for k in KS:
        for kind in ("random", "all_zero", "extreme_finite", "round_even_boundaries"):
            x = make_case(kind, k)
            report["correctness"].append(validate(x, kind, k))
            del x
            gc.collect()
            torch.cuda.empty_cache()

    correctness_pass = all(
        item["q_bitwise_equal"] and item["scale_bitwise_equal"]
        for item in report["correctness"]
    )
    report["correctness_pass"] = correctness_pass
    if not correctness_pass:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise RuntimeError("bitwise correctness gate failed")

    for k in KS:
        x = make_case("random", k)
        report["timing"].append(benchmark(x, k))
        del x
        gc.collect()
        torch.cuda.empty_cache()

    by_k = {item["shape"][1]: item for item in report["timing"]}
    predicted_saved_ms = (
        2000 * by_k[5376]["saved_ms_per_call"]
        + 1000 * by_k[7168]["saved_ms_per_call"]
        + 1000 * by_k[14336]["saved_ms_per_call"]
    )
    report["prediction"] = {
        "full20_saved_ms": predicted_saved_ms,
        "full20_saved_seconds": predicted_saved_ms / 1000.0,
        "performance_gate_pass": predicted_saved_ms > 1000.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
