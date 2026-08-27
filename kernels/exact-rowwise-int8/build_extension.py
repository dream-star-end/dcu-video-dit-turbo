from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx936")

flags = [
    "-O3",
    "-fno-fast-math",
    "-fno-unsafe-math-optimizations",
    "-ffp-contract=off",
]
module = load(
    name="h3_exact_rowwise_int8",
    sources=[str(ROOT / "quantizer.cpp"), str(ROOT / "quantizer_kernel.hip")],
    build_directory=str(BUILD),
    extra_cflags=flags,
    extra_cuda_cflags=flags,
    with_cuda=True,
    verbose=True,
)
print(module.__file__)
