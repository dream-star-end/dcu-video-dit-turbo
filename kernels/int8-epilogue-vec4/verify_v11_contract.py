from __future__ import annotations

import ast
import importlib.util
from collections import Counter
from pathlib import Path

import torch


LAUNCHER = Path(
    "benchmarks/recovered-runtime/"
    "benchmark_launcher_v11_exact_quant_vec4.py"
)
DSO = Path(
    "kernels/int8-epilogue-vec4/build/"
    "h3_exact_epilogue_vec4.so"
)


class FakeTensor:
    def __init__(self, shape, dtype, *, contiguous=True, device="cuda:0"):
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = True
        self.device = device
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def dim(self):
        return len(self.shape)

    def numel(self):
        value = 1
        for size in self.shape:
            value *= size
        return value


tree = ast.parse(LAUNCHER.read_text())
wanted = {"_load_vec4_epilogue", "_vec4_epilogue_gate"}
functions = [
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in wanted
]
assert {node.name for node in functions} == wanted
namespace = {
    "torch": torch,
    "importlib": importlib,
    "_vec4_epilogue": None,
    "_vec4_epilogue_so": str(DSO),
    "_vec4_epilogue_enabled": True,
    "_VEC4_EPILOGUE_M": 11819,
    "_VEC4_EPILOGUE_N": frozenset((5376, 21504, 28672)),
    "_is_gfx936": lambda _tensor: True,
}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(LAUNCHER), "exec"), namespace)

module = namespace["_load_vec4_epilogue"]()
assert hasattr(module, "epilogue")
namespace["_vec4_epilogue"] = None
namespace["_vec4_epilogue_so"] = str(DSO.with_name("missing_vec4_epilogue.so"))
try:
    namespace["_load_vec4_epilogue"]()
except (ImportError, OSError):
    pass
else:
    raise AssertionError("missing candidate DSO did not hard-fail")
namespace["_vec4_epilogue"] = module
namespace["_vec4_epilogue_so"] = str(DSO)


def gate(acc, row, col, bias=None, out_dtype=torch.bfloat16):
    return namespace["_vec4_epilogue_gate"](acc, row, col, bias, out_dtype)


for n in (5376, 21504, 28672):
    acc = FakeTensor((11819, n), torch.int32)
    row = FakeTensor((11819,), torch.float32)
    col = FakeTensor((n,), torch.float32)
    assert gate(acc, row, col) == (True, "eligible")

acc = FakeTensor((11819, 5376), torch.int32)
row = FakeTensor((11819,), torch.float32)
col = FakeTensor((5376,), torch.float32)
assert gate(acc, row, col, bias=FakeTensor((5376,), torch.bfloat16))[1] == "bias"
assert gate(acc, row, col, out_dtype=torch.float16)[1] == "out_dtype"
assert gate(FakeTensor((11818, 5376), torch.int32), row, col)[1] == "M"
assert gate(FakeTensor((11819, 5377), torch.int32), row, FakeTensor((5377,), torch.float32))[1] == "N"
assert gate(FakeTensor((11819, 5376), torch.float16), row, col)[1] == "acc_dtype"
assert gate(acc, FakeTensor((11819,), torch.bfloat16), col)[1] == "row_scale_dtype"
assert gate(acc, row, FakeTensor((5376,), torch.bfloat16))[1] == "col_scale_dtype"
assert gate(FakeTensor((11819, 5376), torch.int32, contiguous=False), row, col)[1] == "acc_noncontiguous"
assert gate(acc, FakeTensor((11819,), torch.float32, contiguous=False), col)[1] == "row_scale_noncontiguous"
assert gate(acc, row, FakeTensor((5376,), torch.float32, contiguous=False))[1] == "col_scale_noncontiguous"
assert gate(acc, FakeTensor((11819,), torch.float32, device="cuda:1"), col)[1] == "device_mismatch"

namespace["_is_gfx936"] = lambda _tensor: False
assert gate(acc, row, col)[1] == "arch"

print("v11 contract PASS: DSO load + eligible N set + strict fallback reasons")
