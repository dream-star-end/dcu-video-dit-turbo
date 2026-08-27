#!/usr/bin/env python3
"""Bitwise validation for the measured 15 s H3 I2V/FLF INT8 row counts."""

from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path

import torch
from comfy_kitchen.backends.eager import quantization as qops


NEW_M = (12055, 12280)
LINEAR_SHAPES = (
    (5376, 21504),
    (7168, 5376),
    (5376, 28672),
    (14336, 5376),
)
QUANTIZER_SO = Path("kernels/exact-rowwise-int8/build/h3_exact_rowwise_int8.so")
REFERENCE_EPILOGUE_SO = Path("kernels/int8-epilogue/build/h3_hip_epilogue.so")
VEC4_EPILOGUE_SO = Path("kernels/int8-epilogue-vec4/build/h3_exact_epilogue_vec4.so")


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quantizer = load_extension("h3_exact_rowwise_int8", QUANTIZER_SO)
reference_epilogue = load_extension("h3_hip_epilogue", REFERENCE_EPILOGUE_SO)
vec4_epilogue = load_extension("h3_exact_epilogue_vec4", VEC4_EPILOGUE_SO)


def cleanup(*values) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(torch.count_nonzero(left != right).item())


def validate_shape(m: int, k: int, n: int, seed: int, check_epilogue: bool) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    q_ref, scale_ref = qops.quantize_int8_rowwise(x)
    q_new, scale_new = quantizer.quantize_int8_rowwise(x)
    torch.cuda.synchronize()

    quant_mismatch = mismatch_count(q_ref, q_new)
    scale_mismatch = mismatch_count(scale_ref.view(torch.int32), scale_new.view(torch.int32))

    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=generator)
    accumulator_new = torch._int_mm(q_new, weight.T)
    accumulator_ref = qops._int8_matmul_accumulate(q_ref, weight.T)
    torch.cuda.synchronize()
    accumulator_mismatch = mismatch_count(accumulator_ref, accumulator_new)

    epilogue_mismatch = None
    if check_epilogue:
        row_scale = scale_new.reshape(-1).to(torch.float32).contiguous()
        col_scale = (torch.rand((n,), device="cuda", dtype=torch.float32, generator=generator) * 0.02 + 1e-5).contiguous()
        empty_bias = torch.empty((0,), device="cuda", dtype=torch.bfloat16)
        output_ref = reference_epilogue.epilogue(
            accumulator_new, row_scale, col_scale, empty_bias
        )
        output_new = vec4_epilogue.epilogue(accumulator_new, row_scale, col_scale)
        torch.cuda.synchronize()
        epilogue_mismatch = mismatch_count(output_ref, output_new)
        cleanup(output_ref, output_new, row_scale, col_scale, empty_bias)

    result = {
        "shape": [m, k, n],
        "quantized_value_mismatch": quant_mismatch,
        "quantized_scale_bit_mismatch": scale_mismatch,
        "int32_accumulator_mismatch": accumulator_mismatch,
        "vec4_epilogue_bf16_mismatch": epilogue_mismatch,
    }
    cleanup(x, q_ref, scale_ref, q_new, scale_new, weight, accumulator_new, accumulator_ref)
    return result


def validate_round_even(m: int) -> dict:
    k = 5376
    pattern = torch.tensor(
        [63.5, -63.5, 62.5, 1.25, 0.75, 0.25, -0.25, -0.75, -1.25, 0.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    count = m * k
    x = pattern.repeat((count + pattern.numel() - 1) // pattern.numel())[:count]
    x = x.reshape(m, k).contiguous()
    q_ref, scale_ref = qops.quantize_int8_rowwise(x)
    q_new, scale_new = quantizer.quantize_int8_rowwise(x)
    torch.cuda.synchronize()
    result = {
        "shape": [m, k],
        "quantized_value_mismatch": mismatch_count(q_ref, q_new),
        "quantized_scale_bit_mismatch": mismatch_count(
            scale_ref.view(torch.int32), scale_new.view(torch.int32)
        ),
    }
    cleanup(x, pattern, q_ref, scale_ref, q_new, scale_new)
    return result


def main() -> None:
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx936":
        raise RuntimeError("validation requires a gfx936 device")
    started = time.perf_counter()
    results = []
    for m_index, m in enumerate(NEW_M):
        epilogue_ns = set()
        for shape_index, (k, n) in enumerate(LINEAR_SHAPES):
            check_epilogue = n not in epilogue_ns
            results.append(
                validate_shape(
                    m, k, n, 2026081200 + m_index * 100 + shape_index,
                    check_epilogue,
                )
            )
            epilogue_ns.add(n)
    round_even = [validate_round_even(m) for m in NEW_M]
    fields = (
        "quantized_value_mismatch",
        "quantized_scale_bit_mismatch",
        "int32_accumulator_mismatch",
        "vec4_epilogue_bf16_mismatch",
    )
    passed = all(
        item.get(field) in (None, 0)
        for item in results
        for field in fields
    ) and all(
        item["quantized_value_mismatch"] == 0
        and item["quantized_scale_bit_mismatch"] == 0
        for item in round_even
    )
    report = {
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "round_even_results": round_even,
        "bitwise_pass": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
