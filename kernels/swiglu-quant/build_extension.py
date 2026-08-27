from pathlib import Path
import os
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
build = root / "build"
build.mkdir(exist_ok=True)
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx936")
flags = ["-O3", "-fno-fast-math", "-fno-unsafe-math-optimizations", "-ffp-contract=off"]
module = load(
    name="h3_exact_swiglu_quant",
    sources=[str(root / "swiglu_quant.cpp"), str(root / "swiglu_quant_kernel.hip")],
    build_directory=str(build),
    extra_cflags=flags,
    extra_cuda_cflags=flags,
    with_cuda=True,
    verbose=True,
)
print(module.__file__)
