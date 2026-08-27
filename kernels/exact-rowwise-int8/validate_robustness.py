from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import torch
from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise


root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "h3_exact_rowwise_int8", root / "build" / "h3_exact_rowwise_int8.so"
)
extension = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extension)


def compare(x: torch.Tensor, label: str) -> dict:
    q_ref, s_ref = quantize_int8_rowwise(x)
    q_new, s_new = extension.quantize_int8_rowwise(x)
    torch.cuda.synchronize()
    result = {
        "label": label,
        "shape": list(x.shape),
        "q_mismatch": int(torch.count_nonzero(q_ref != q_new).item()),
        "scale_bit_mismatch": int(
            torch.count_nonzero(s_ref.view(torch.int32) != s_new.view(torch.int32)).item()
        ),
    }
    del q_ref, s_ref, q_new, s_new, x
    gc.collect()
    torch.cuda.empty_cache()
    return result


results = []
for k in (5376, 7168, 14336):
    for seed in (20260813, 20260814):
        generator = torch.Generator(device="cuda").manual_seed(seed + k)
        x = torch.randn(
            (11819, k), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        results.append(compare(x, f"random_seed_{seed}"))

# A non-unit exact scale: max=63.5 -> FP32 scale=0.5. Values such as 1.25
# therefore exercise exact half-integer rounding after BF16 division.
pattern = torch.tensor(
    [63.5, -63.5, 62.5, 1.25, 0.75, 0.25, -0.25, -0.75, -1.25, 0.0],
    device="cuda",
    dtype=torch.bfloat16,
)
for k in (5376, 7168, 14336):
    count = 11819 * k
    x = pattern.repeat((count + pattern.numel() - 1) // pattern.numel())[:count]
    results.append(compare(x.reshape(11819, k).contiguous(), "round_even_scale_0.5"))

nonfinite = torch.tensor(
    [[float("nan"), float("inf"), float("-inf"), 1.0, -1.0, 0.0]],
    device="cuda",
    dtype=torch.bfloat16,
)
results.append(compare(nonfinite, "nan_and_infinities"))

report = {
    "results": results,
    "pass": all(x["q_mismatch"] == 0 and x["scale_bit_mismatch"] == 0 for x in results),
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["pass"]:
    raise SystemExit(1)
