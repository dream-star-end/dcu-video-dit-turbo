from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SO_PATHS = sorted((ROOT / "build").glob("h3_hip_epilogue*.so"))
if not SO_PATHS:
    raise RuntimeError("extension not built; run build_extension.py first")
spec = importlib.util.spec_from_file_location("h3_hip_epilogue", SO_PATHS[-1])
extension = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extension)


# The full-shape profile records orig M=8216, padded to M=8224 only inside
# _int8_matmul_accumulate. The current epilogue runs after slicing back to 8216.
# These are the four quantized DiT linears: qkv, out/fc2, and fc1.
SHAPES = [
    (8216, 21504, "qkv"),
    (8216, 5376, "out_or_fc2"),
    (8216, 28672, "fc1"),
]


def eager_epilogue(acc, row_scale, col_scale, bias):
    # Literal ordering from comfy_kitchen.backends.eager.quantization.int8_linear.
    m, n = acc.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    scaled_parts = []
    weight_scale = col_scale.reshape(1, -1)
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        chunk = acc[start:stop].float()
        chunk_scales = row_scale[start:stop].to(torch.float32) * weight_scale
        chunk_scaled = chunk * chunk_scales
        scaled_parts.append(chunk_scaled.to(torch.bfloat16))
    result = torch.cat(scaled_parts, dim=0)
    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    return result


def event_ms(callable_, warmup=8, repeats=20, rounds=5):
    samples = []
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            callable_()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / repeats)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def peak_incremental_bytes(callable_):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = callable_()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    output_bytes = output.numel() * output.element_size()
    result = {
        "incremental_peak_bytes": int(peak - baseline),
        "output_bytes": int(output_bytes),
    }
    del output
    torch.cuda.empty_cache()
    return result


def check_one(m, n, bias_dtype, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    acc = torch.randint(
        -2_000_000,
        2_000_001,
        (m, n),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    row = torch.rand((m, 1), dtype=torch.float32, device="cuda", generator=generator) * 0.02 + 1e-5
    col = torch.rand((n,), dtype=torch.float32, device="cuda", generator=generator) * 0.02 + 1e-5
    if bias_dtype is None:
        bias = torch.empty((0,), dtype=torch.bfloat16, device="cuda")
        bias_ref = None
        label = "none"
    else:
        bias = torch.randn((n,), dtype=bias_dtype, device="cuda", generator=generator)
        bias_ref = bias
        label = str(bias_dtype).replace("torch.", "")

    expected = eager_epilogue(acc, row, col, bias_ref)
    actual = extension.epilogue(acc, row, col, bias)
    torch.cuda.synchronize()
    equal = torch.equal(expected, actual)
    diff = (expected.float() - actual.float()).abs()
    return {
        "bias": label,
        "seed": seed,
        "equal": equal,
        "max_abs": float(diff.max().item()),
        "mismatch_count": int(torch.count_nonzero(expected != actual).item()),
    }, (acc, row, col, bias, bias_ref)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("HIP device is not available")
    props = torch.cuda.get_device_properties(0)
    report = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": props.name,
        "arch": props.gcnArchName,
        "compile_flags": [
            "-O3",
            "-fno-fast-math",
            "-fno-unsafe-math-optimizations",
            "-ffp-contract=off",
        ],
        "correctness": [],
        "timing": [],
    }

    cached_inputs = {}
    for m, n, name in SHAPES:
        # The four production DiT INT8 linears have no bias. Use three full-shape
        # seeds for that path, then exercise both optional-bias dtypes once.
        for seed in (20260810, 20260811, 20260812):
            result, tensors = check_one(m, n, None, seed)
            result.update({"shape": [m, n], "linear": name})
            report["correctness"].append(result)
            if seed == 20260810:
                cached_inputs[(m, n, name, "none")] = tensors
        for bias_dtype in (torch.bfloat16, torch.float32):
            result, _ = check_one(m, n, bias_dtype, 20260810)
            result.update({"shape": [m, n], "linear": name})
            report["correctness"].append(result)

    if not all(item["equal"] for item in report["correctness"]):
        print(json.dumps(report, indent=2, sort_keys=True))
        raise RuntimeError("strict correctness gate failed")

    for m, n, name in SHAPES:
        acc, row, col, bias, bias_ref = cached_inputs[(m, n, name, "none")]
        fused_call = lambda: extension.epilogue(acc, row, col, bias)
        eager_call = lambda: eager_epilogue(acc, row, col, bias_ref)
        fused = event_ms(fused_call)
        eager = event_ms(eager_call)
        report["timing"].append({
            "shape": [m, n],
            "linear": name,
            "bias": "none",
            "fused": fused,
            "eager": eager,
            "speedup": eager["median_ms"] / fused["median_ms"],
            "saved_ms": eager["median_ms"] - fused["median_ms"],
            "peak": {
                "fused": peak_incremental_bytes(fused_call),
                "eager": peak_incremental_bytes(eager_call),
            },
        })

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
