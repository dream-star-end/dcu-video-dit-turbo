from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx936")

module = load(
    name="h3_hip_epilogue",
    sources=[str(ROOT / "epilogue.cpp"), str(ROOT / "epilogue_kernel.hip")],
    build_directory=str(BUILD),
    extra_cflags=[
        "-O3",
        "-fno-fast-math",
        "-fno-unsafe-math-optimizations",
        "-ffp-contract=off",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-fno-fast-math",
        "-fno-unsafe-math-optimizations",
        "-ffp-contract=off",
    ],
    with_cuda=True,
    verbose=True,
)

print(module.__file__)
