# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    # "fp16": "Float16",
    # "bf16": "BFloat16",
    "e4m3": "Float8_e4m3_t"
}

HEAD_DIMENSIONS = [[32,32], [64,64], [96,96], [128,128], [192,128]]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {QK_HEAD_DIM}, {V_HEAD_DIM}>(Flash_fwd_params &params, hipStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch_<{DTYPE}, {QK_HEAD_DIM}, {V_HEAD_DIM}>(Flash_fwd_params &params, hipStream_t stream);
"""

KERNEL_IMPL_TEMPLATE_BWD_1HEAD = """#include "flash_bwd_launch_template.h"

#if defined(BUILD_FA_BWD) {JUDGEMENT}
template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {HEAD_DIM}>(Flash_bwd_params &params, hipStream_t stream, const bool configure) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream, configure);
}}
#endif
"""

KERNEL_IMPL_TEMPLATE_BWD_2HEAD = """#include "flash_bwd_launch_template.h"
#if defined(BUILD_FA_BWD) {JUDGEMENT}
template<>
void run_mha_bwd_<{DTYPE}, {QK_HEAD_DIM}, {V_HEAD_DIM}>(Flash_bwd_params &params, hipStream_t stream, const bool configure) {{
    run_mha_bwd_qkhdim{QK_HEAD_DIM}_vhdim{V_HEAD_DIM}<{DTYPE}>(params, stream, configure);
}}
#endif
"""

@dataclass
class Kernel:
    dtype: str
    qk_head_dim: int
    v_head_dim: int
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], QK_HEAD_DIM=self.qk_head_dim, V_HEAD_DIM=self.v_head_dim
            )
        elif self.direction == "bwd":
            judgement = "&& !defined(HEADDIM_128_ONLY)"
            if self.qk_head_dim == self.v_head_dim:
                if self.qk_head_dim == 128:
                    judgement = ""
                return KERNEL_IMPL_TEMPLATE_BWD_1HEAD.format(
                    DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.qk_head_dim, JUDGEMENT = judgement
                )
            else:
                return KERNEL_IMPL_TEMPLATE_BWD_2HEAD.format(
                    DTYPE=DTYPE_MAP[self.dtype], QK_HEAD_DIM=self.qk_head_dim, V_HEAD_DIM=self.v_head_dim, JUDGEMENT = judgement
                )
        else:
            return KERNEL_IMPL_TEMPLATE_FWD_SPLIT.format(
                DTYPE=DTYPE_MAP[self.dtype], QK_HEAD_DIM=self.qk_head_dim, V_HEAD_DIM=self.v_head_dim
            )

    @property
    def filename(self) -> str:
        if self.qk_head_dim == self.v_head_dim:
            return f"flash_{self.direction}_hdim{self.qk_head_dim}_{self.dtype}.cpp"
        else:
            return f"flash_{self.direction}_qkhdim{self.qk_head_dim}_vhdim{self.v_head_dim}_{self.dtype}.cpp"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS):
        # for direction in ["fwd", "bwd", "fwd_split"]:
        for direction in ["bwd"]:
            yield Kernel(dtype=dtype, qk_head_dim=head_dim[0], v_head_dim=head_dim[1], direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
