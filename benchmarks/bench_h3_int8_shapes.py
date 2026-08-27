#!/usr/bin/env python3
"""Benchmark the exact MiniMax H3 INT8 linear shapes on one HIP device.

The script keeps weight storage in the checkpoint layout [N, K] and measures
both the zero-copy transpose view and a row-major materialization.  It also
checks that every BLAS/backend candidate returns identical INT32 accumulators.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import time

import torch

import comfy_kitchen.backends.eager.quantization as qops
import comfy_kitchen.backends.cuda as cudaops


SHAPES = {
    "qkv": (11819, 5376, 21504),
    "out": (11819, 7168, 5376),
    "fc1": (11819, 5376, 28672),
    "fc2": (11819, 14336, 5376),
}


def synchronize() -> None:
    torch.cuda.synchronize()


def timed(fn, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        value = fn()
    synchronize()
    del value
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        del value
    return statistics.median(samples), samples


def load_epilogue(path: str):
    # The pybind module is compiled with this exact initialization symbol.
    spec = importlib.util.spec_from_file_location("h3_hip_epilogue", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load epilogue: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_backend(name: str) -> str:
    torch.backends.cuda.preferred_blas_library(name)
    return str(torch._C._get_blas_preferred_backend())


def tensor_signature(value: torch.Tensor) -> dict[str, int]:
    flat = value.reshape(-1)
    stride = max(1, flat.numel() // 4096)
    sample = flat[::stride].to(torch.int64)
    # This is only a compact diagnostic; strict equality is checked separately.
    return {
        "numel": flat.numel(),
        "sum_sample": int(sample.sum().item()),
        "abs_sum_sample": int(sample.abs().sum().item()),
    }


def bench_shape(
    name: str,
    shape: tuple[int, int, int],
    backends: list[str],
    warmup: int,
    repeats: int,
    epilogue,
) -> dict:
    m, k, n = shape
    torch.manual_seed(20260812)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand((n,), device="cuda", dtype=torch.float32) * 0.02 + 1e-4
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    synchronize()

    qx, x_scale = qops.quantize_int8_rowwise(x)
    b_view = weight.T
    b_contig = b_view.contiguous()
    synchronize()

    quant_ms, quant_samples = timed(
        lambda: qops.quantize_int8_rowwise(x), warmup, repeats
    )
    copy_ms, copy_samples = timed(lambda: weight.T.contiguous(), warmup, repeats)

    result = {
        "shape": {"m": m, "k": k, "n": n},
        "quantize_ms": quant_ms,
        "quantize_samples_ms": quant_samples,
        "transpose_copy_ms": copy_ms,
        "transpose_copy_samples_ms": copy_samples,
        "candidates": {},
    }
    reference = None
    reference_backend = None
    for backend in backends:
        try:
            selected = set_backend(backend)
            synchronize()
            out_view = torch._int_mm(qx, b_view)
            synchronize()
            view_ms, view_samples = timed(
                lambda: torch._int_mm(qx, b_view), warmup, repeats
            )
            contig_ms, contig_samples = timed(
                lambda: torch._int_mm(qx, b_contig), warmup, repeats
            )
            if reference is None:
                reference = out_view.clone()
                reference_backend = backend
                exact = True
                max_abs = 0
            else:
                exact = bool(torch.equal(reference, out_view))
                max_abs = int((reference - out_view).abs().max().item())
            epilogue_ms, epilogue_samples = timed(
                lambda: epilogue.epilogue(out_view, x_scale.reshape(-1), weight_scale, bias),
                warmup,
                repeats,
            )

            def full_pipeline():
                qx_full, xs_full = qops.quantize_int8_rowwise(x)
                accum = torch._int_mm(qx_full, b_view)
                return epilogue.epilogue(accum, xs_full.reshape(-1), weight_scale, bias)

            full_ms, full_samples = timed(full_pipeline, warmup, repeats)
            result["candidates"][backend] = {
                "selected": selected,
                "view_mm_ms": view_ms,
                "view_mm_samples_ms": view_samples,
                "contiguous_mm_ms": contig_ms,
                "contiguous_mm_samples_ms": contig_samples,
                "epilogue_ms": epilogue_ms,
                "epilogue_samples_ms": epilogue_samples,
                "full_ms": full_ms,
                "full_samples_ms": full_samples,
                "exact_vs_reference": exact,
                "max_abs_vs_reference": max_abs,
                "signature": tensor_signature(out_view),
            }
            del out_view
        except Exception as exc:
            result["candidates"][backend] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
        synchronize()
        torch.cuda.empty_cache()

    # The bundled native extension exposes a raw cuBLAS/HIPBLAS entry point
    # that consumes checkpoint-native weight storage [N, K].  This is the same
    # integer GEMM as torch._int_mm(qx, weight.T), without a strided RHS or a
    # persistent transposed copy.
    try:
        stream_ptr = torch.cuda.current_stream(qx.device).cuda_stream
        workspace = cudaops.get_cublas_workspace()

        def native_mm():
            out = torch.empty((m, n), dtype=torch.int32, device=qx.device)
            cudaops._C.cublas_gemm_int8(
                qx,
                weight,
                out,
                workspace,
                stream_ptr,
            )
            return out

        native_out = native_mm()
        synchronize()
        native_ms, native_samples = timed(native_mm, warmup, repeats)
        exact = bool(torch.equal(reference, native_out))
        max_abs = int((reference - native_out).abs().max().item())
        native_epilogue_ms, native_epilogue_samples = timed(
            lambda: epilogue.epilogue(
                native_out, x_scale.reshape(-1), weight_scale, bias
            ),
            warmup,
            repeats,
        )

        def native_full_pipeline():
            qx_full, xs_full = qops.quantize_int8_rowwise(x)
            accum = torch.empty((m, n), dtype=torch.int32, device=x.device)
            cudaops._C.cublas_gemm_int8(
                qx_full,
                weight,
                accum,
                workspace,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
            return epilogue.epilogue(
                accum, xs_full.reshape(-1), weight_scale, bias
            )

        native_full_ms, native_full_samples = timed(
            native_full_pipeline, warmup, repeats
        )
        result["candidates"]["native_cublas"] = {
            "selected": "comfy_kitchen.cuda._C.cublas_gemm_int8",
            "view_mm_ms": native_ms,
            "view_mm_samples_ms": native_samples,
            "epilogue_ms": native_epilogue_ms,
            "epilogue_samples_ms": native_epilogue_samples,
            "full_ms": native_full_ms,
            "full_samples_ms": native_full_samples,
            "exact_vs_reference": exact,
            "max_abs_vs_reference": max_abs,
            "signature": tensor_signature(native_out),
        }
        del native_out
    except Exception as exc:
        result["candidates"]["native_cublas"] = {
            "error": f"{type(exc).__name__}: {exc}"
        }
    synchronize()
    torch.cuda.empty_cache()
    result["reference_backend"] = reference_backend
    del reference, qx, x_scale, b_view, b_contig, x, weight, weight_scale, bias
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--backends", nargs="+", default=["hipblas", "hipblaslt"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--epilogue-so",
        default="kernels/int8-epilogue/build/h3_hip_epilogue.so",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HIP/CUDA device is unavailable")
    epilogue = load_epilogue(args.epilogue_so)
    payload = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "results": {},
    }
    started = time.time()
    for name in args.shapes:
        payload["results"][name] = bench_shape(
            name,
            SHAPES[name],
            args.backends,
            args.warmup,
            args.repeats,
            epilogue,
        )
        print(json.dumps({name: payload["results"][name]}, ensure_ascii=False), flush=True)
    payload["wall_seconds"] = time.time() - started
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
