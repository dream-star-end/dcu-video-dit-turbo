from __future__ import annotations

import gc
import importlib.util
import json
import statistics
from pathlib import Path

import torch
from comfy_kitchen.backends._activations import apply_input_act


root = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fused = load("h3_exact_swiglu_quant", root / "build" / "h3_exact_swiglu_quant.so")
quant = load(
    "h3_exact_rowwise_int8",
    Path("kernels/exact-rowwise-int8/build/h3_exact_rowwise_int8.so"),
)
M, K = 11819, 14336


def baseline(x):
    return quant.quantize_int8_rowwise(apply_input_act(x, "swiglu"))


def candidate(x):
    return fused.swiglu_quantize_int8_rowwise(x)


def compare(x, label):
    qr, sr = baseline(x)
    qn, sn = candidate(x)
    torch.cuda.synchronize()
    result = {
        "label": label,
        "shape": list(x.shape),
        "q_mismatch": int(torch.count_nonzero(qr != qn).item()),
        "scale_bit_mismatch": int(
            torch.count_nonzero(sr.view(torch.int32) != sn.view(torch.int32)).item()
        ),
    }
    del qr, sr, qn, sn, x
    gc.collect()
    torch.cuda.empty_cache()
    return result


def timed(fn, x, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(x)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


correctness = []
for seed in (20260812, 20260813, 20260814):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((M, 2 * K), device="cuda", dtype=torch.bfloat16, generator=generator)
    correctness.append(compare(x, f"full_random_{seed}"))

x = torch.zeros((M, 2 * K), device="cuda", dtype=torch.bfloat16)
correctness.append(compare(x, "full_zero"))

finfo = torch.finfo(torch.bfloat16)
pattern = torch.tensor(
    [finfo.max, -finfo.max, finfo.tiny, -finfo.tiny, 0.0, -0.0, 127.0,
     -127.0, 2.5, -2.5, 1.5, -1.5, 0.5, -0.5],
    device="cuda", dtype=torch.bfloat16,
)
count = 257 * 2 * K
x = pattern.repeat((count + pattern.numel() - 1) // pattern.numel())[:count].reshape(257, 2 * K)
correctness.append(compare(x.contiguous(), "finite_extremes_and_boundaries"))

if not all(x["q_mismatch"] == 0 and x["scale_bit_mismatch"] == 0 for x in correctness):
    print(json.dumps({"correctness": correctness, "pass": False}, indent=2))
    raise SystemExit(1)

generator = torch.Generator(device="cuda").manual_seed(20260815)
x = torch.randn((M, 2 * K), device="cuda", dtype=torch.bfloat16, generator=generator)
for _ in range(12):
    baseline(x)
    candidate(x)
torch.cuda.synchronize()
base_samples, fused_samples = [], []
for index in range(11):
    if index % 2 == 0:
        base_samples.append(timed(baseline, x, 10))
        fused_samples.append(timed(candidate, x, 10))
    else:
        fused_samples.append(timed(candidate, x, 10))
        base_samples.append(timed(baseline, x, 10))

base_median = statistics.median(base_samples)
fused_median = statistics.median(fused_samples)
saved = base_median - fused_median
report = {
    "correctness": correctness,
    "correctness_pass": True,
    "timing": {
        "baseline_samples_ms": base_samples,
        "candidate_samples_ms": fused_samples,
        "baseline_median_ms": base_median,
        "candidate_median_ms": fused_median,
        "speedup": base_median / fused_median,
        "saved_ms_per_block": saved,
        "predicted_full20_saved_seconds": saved * 50 * 20 / 1000,
    },
}
print(json.dumps(report, indent=2, sort_keys=True))
