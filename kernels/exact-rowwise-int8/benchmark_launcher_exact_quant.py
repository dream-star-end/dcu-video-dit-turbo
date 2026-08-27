"""Launch ComfyUI with quality-equivalent gfx936 INT8 optimizations.

The candidate combines the verified no-copy INT8 weight transpose path with a
scratch-built HIP epilogue that performs the exact eager FP32 scale ordering and
BF16 rounding in one kernel.  It does not modify the installed runtime.
"""

from __future__ import annotations

import logging
import importlib.util
import atexit
import os
import runpy
import signal
import sys
from collections import Counter

COMFY_ROOT = os.environ.get("H3_COMFY_ROOT", "/path/to/ComfyUI")
sys.path.insert(0, COMFY_ROOT)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

try:
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    _tv_lib = None

import comfy_kitchen.backends.eager as _eager  # noqa: E402
import comfy_kitchen.backends.eager.quantization as _q  # noqa: E402


if os.environ.get("H3_BENCH_NONCE_INPUT", "0") == "1":
    # Add an ignored cache key to RandomNoise.  Varying this value reruns only
    # noise/sampling/decoding while model loaders and text conditioning remain
    # cached, giving a true hot same-seed A/B without changing generated noise.
    import comfy_extras.nodes_custom_sampler as _custom_sampler  # noqa: E402

    _OriginalRandomNoise = _custom_sampler.RandomNoise
    _io = _custom_sampler.io

    class _BenchmarkRandomNoise(_OriginalRandomNoise):
        @classmethod
        def define_schema(cls):
            return _io.Schema(
                node_id="RandomNoise",
                category="model/sampling/noise",
                inputs=[
                    _io.Int.Input(
                        "noise_seed",
                        default=0,
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        control_after_generate=True,
                    ),
                    _io.Int.Input(
                        "benchmark_nonce",
                        default=0,
                        min=0,
                        max=0x7FFFFFFF,
                        optional=True,
                    ),
                ],
                outputs=[_io.Noise.Output()],
            )

        @classmethod
        def execute(cls, noise_seed, benchmark_nonce=0):
            del benchmark_nonce
            return _OriginalRandomNoise.execute(noise_seed)

        get_noise = execute

    _custom_sampler.RandomNoise = _BenchmarkRandomNoise


_original_int8_linear = _eager.int8_linear
_no_copy_enabled = os.environ.get("H3_INT8_NOCOPY", "0") == "1"
_epilogue_enabled = os.environ.get("H3_INT8_EPILOGUE", "0") == "1"
_epilogue_so = os.environ.get(
    "H3_INT8_EPILOGUE_SO",
    "kernels/int8-epilogue/build/h3_hip_epilogue.so",
)
_epilogue = None
_empty_bias_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

# Experimental, strictly gated gfx936 rowwise quantizer.  Loading is eager on
# purpose: when explicitly enabled, a missing/incompatible DSO must abort the
# launcher rather than silently changing the benchmark back to eager.
_exact_quantizer_enabled = os.environ.get("H3_INT8_EXACT_QUANTIZER", "1") == "1"
_exact_quantizer_so = os.environ.get(
    "H3_INT8_EXACT_QUANTIZER_SO",
    "kernels/exact-rowwise-int8/build/h3_exact_rowwise_int8.so",
)
_exact_quantizer = None
_exact_quantizer_hits = 0
_exact_quantizer_fallbacks = Counter()
_EXACT_QUANTIZER_M = 11819
_EXACT_QUANTIZER_K = frozenset((5376, 7168, 14336))


def _load_exact_quantizer():
    global _exact_quantizer
    if _exact_quantizer is not None:
        return _exact_quantizer
    spec = importlib.util.spec_from_file_location(
        "h3_exact_rowwise_int8", _exact_quantizer_so
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load required H3 exact INT8 quantizer from {_exact_quantizer_so}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "quantize_int8_rowwise"):
        raise RuntimeError(
            f"H3 exact INT8 quantizer DSO has no quantize_int8_rowwise symbol: {_exact_quantizer_so}"
        )
    _exact_quantizer = module
    return module


def _exact_quantizer_gate(x: torch.Tensor, stochastic_rounding: int) -> tuple[bool, str]:
    if not _exact_quantizer_enabled:
        return False, "disabled"
    if stochastic_rounding != 0:
        return False, "stochastic_rounding"
    if x.dtype != torch.bfloat16:
        return False, "dtype"
    if not x.is_cuda:
        return False, "device"
    if not x.is_contiguous():
        return False, "noncontiguous"
    if x.dim() != 2 or x.shape[0] != _EXACT_QUANTIZER_M:
        return False, "M"
    if x.shape[1] not in _EXACT_QUANTIZER_K:
        return False, "K"
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
    if arch != "gfx936":
        return False, "arch"
    return True, "eligible"


def _quantize_activation(x: torch.Tensor, stochastic_rounding: int = 0):
    global _exact_quantizer_hits
    eligible, reason = _exact_quantizer_gate(x, stochastic_rounding)
    if eligible:
        _exact_quantizer_hits += 1
        if _exact_quantizer_hits == 1 or _exact_quantizer_hits % 500 == 0:
            logging.warning(
                "H3 exact INT8 quantizer hits=%d fallbacks=%d",
                _exact_quantizer_hits,
                sum(_exact_quantizer_fallbacks.values()),
            )
        return _exact_quantizer.quantize_int8_rowwise(x)

    _exact_quantizer_fallbacks[reason] += 1
    if _exact_quantizer_fallbacks[reason] == 1:
        logging.warning(
            "H3 exact INT8 quantizer fallback reason=%s shape=%s dtype=%s contiguous=%s",
            reason,
            tuple(x.shape),
            x.dtype,
            x.is_contiguous(),
        )
    return _q.quantize_int8_rowwise(
        x, stochastic_rounding=stochastic_rounding
    )


def _log_exact_quantizer_counts():
    logging.warning(
        "H3 exact INT8 quantizer final hits=%d fallbacks=%d reasons=%s",
        _exact_quantizer_hits,
        sum(_exact_quantizer_fallbacks.values()),
        dict(sorted(_exact_quantizer_fallbacks.items())),
    )


if _exact_quantizer_enabled:
    _load_exact_quantizer()  # Hard-fail at process startup if the DSO is unusable.
    logging.warning("H3 exact INT8 quantizer loaded: %s", _exact_quantizer_so)
atexit.register(_log_exact_quantizer_counts)


def _load_epilogue():
    global _epilogue
    if _epilogue is not None:
        return _epilogue
    spec = importlib.util.spec_from_file_location("h3_hip_epilogue", _epilogue_so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load H3 HIP epilogue from {_epilogue_so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _epilogue = module
    return module


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    value = _empty_bias_cache.get(key)
    if value is None:
        value = torch.empty((0,), device=device, dtype=dtype)
        _empty_bias_cache[key] = value
    return value


def _int8_linear_nocopy(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """Eager INT8 linear without materializing weight.T on every invocation.

    DTK's torch._int_mm accepts the column-major transpose view directly. This
    retains the exact INT8 operands and accumulation while removing a full
    device-to-device copy of every quantized weight matrix on every denoise step.
    """
    x = _q._apply_input_act(x, input_act)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = _q._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _q._rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    # INT8 linear currently requests deterministic quantization.  Keep the
    # stochastic value explicit so the DSO gate cannot accidentally broaden.
    x_8, x_scale = _quantize_activation(x_2d, stochastic_rounding=0)

    # The only intentional change from comfy-kitchen 0.2.26 eager.int8_linear:
    # pass the transpose view rather than allocating weight.T.contiguous().
    result = _q._int8_matmul_accumulate(x_8, weight.T)

    if _epilogue_enabled and out_dtype == torch.bfloat16:
        fused_weight_scale = weight_scale.reshape(-1)
        if fused_weight_scale.numel() == 1:
            fused_weight_scale = fused_weight_scale.expand(result.shape[1]).contiguous()
        else:
            fused_weight_scale = fused_weight_scale.contiguous()
        fused_bias = (
            _empty_bias(result.device, out_dtype)
            if bias is None
            else bias.to(device=result.device).reshape(-1).contiguous()
        )
        if fused_bias.dtype not in (torch.bfloat16, torch.float32):
            fused_bias = fused_bias.to(torch.bfloat16)
        result = _load_epilogue().epilogue(
            result,
            x_scale.reshape(-1).contiguous(),
            fused_weight_scale,
            fused_bias,
        )
    else:
        m, n = result.shape
        chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
        weight_scale_2d = weight_scale.reshape(1, -1)
        scaled_parts = []
        for i in range(0, m, chunk_size):
            end_i = min(i + chunk_size, m)
            chunk = result[i:end_i].float()
            chunk_scales = x_scale[i:end_i].to(device=chunk.device, dtype=torch.float32) * weight_scale_2d
            scaled_parts.append((chunk * chunk_scales).to(out_dtype))
        result = torch.cat(scaled_parts, dim=0)

        if bias is not None:
            result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _int8_linear_dispatch(*args, **kwargs):
    if _no_copy_enabled:
        return _int8_linear_nocopy(*args, **kwargs)
    return _original_int8_linear(*args, **kwargs)


def _set_nocopy(enabled: bool):
    global _no_copy_enabled
    _no_copy_enabled = enabled
    logging.warning("H3 INT8 no-copy path %s", "ENABLED" if enabled else "DISABLED")


def _enable_nocopy(_signum, _frame):
    _set_nocopy(True)


def _disable_nocopy(_signum, _frame):
    _set_nocopy(False)


# Registry dispatch resolves the implementation dynamically from this module.
_eager.int8_linear = _int8_linear_dispatch
_q.int8_linear = _int8_linear_dispatch
signal.signal(signal.SIGUSR1, _enable_nocopy)
signal.signal(signal.SIGUSR2, _disable_nocopy)

sys.argv[0] = os.path.join(COMFY_ROOT, "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")
