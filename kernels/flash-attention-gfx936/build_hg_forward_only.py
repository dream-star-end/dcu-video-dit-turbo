#!/usr/bin/env python3
"""Build only the gfx936 BF16/head-dim-128 forward HG FlashAttention module."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import shlex
import setuptools


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "csrc" / "flash_attn_hipc"
BUILD = ROOT / "build" / os.environ.get(
    "H3_HG_BUILD_NAME",
    "flash_attn_hg_forward_gfx936",
)


def adapt_flags_for_dcc_25_10(namespace: dict, descriptor: dict) -> None:
    """Translate the repository's newer aicc-only flags for DTK 26.04 dcc."""
    original = shlex.split(descriptor["compile_flags"])
    unsupported_llvm_options = {
        "-disable-code-sink",
        "-allow-gvn-convergent-call=true",
        "-disallow-uniform-vmed3-combine=true",
        "-hcu-pre-emit-load-store-opt=false",
    }
    adapted: list[str] = []
    removed: list[str] = []
    index = 0
    while index < len(original):
        token = original[index]
        if (
            token == "-mllvm"
            and index + 1 < len(original)
            and original[index + 1] in unsupported_llvm_options
        ):
            removed.extend(original[index : index + 2])
            index += 2
            continue
        if token == "-Xarch_gfx936" and index + 1 < len(original):
            if original[index + 1] == "-mllvm=-support-768-vgprs=true":
                removed.extend(original[index : index + 2])
                adapted.append("-mllvm=-enable-num-vgprs-768=true")
                index += 2
                continue
        adapted.append(token)
        index += 1

    block_n = int(os.environ.get("H3_HG_BLOCK_N", "128"))
    if block_n not in {64, 96, 128}:
        raise ValueError(f"Unsupported H3_HG_BLOCK_N={block_n}")
    adapted.extend(["-DH3_STANDARD_FWD_ONLY", f"-DH3_BLOCK_N={block_n}"])
    packed_shims = os.environ.get("H3_HG_PACKED", "0") == "1"
    if packed_shims:
        adapted.append("-DH3_PACKED_F32_SHIMS")
    descriptor["compile_flags"] = namespace["_ninja_shell_join"](adapted)
    print("removed_incompatible_flags=" + " ".join(removed))
    print(
        "added_compatible_flags=-mllvm=-enable-num-vgprs-768=true "
        f"-DH3_STANDARD_FWD_ONLY -DH3_BLOCK_N={block_n} "
        + ("-DH3_PACKED_F32_SHIMS" if packed_shims else "")
    )


def select_h3_sources(namespace: dict, descriptor: dict) -> None:
    sources = [
        str(SRC / "h3_flash_api.cpp"),
        str(SRC / "src" / "target" / "flash_fwd_hdim128_bf16.cpp"),
    ]
    descriptor["sources"] = sources
    descriptor["objects"] = [
        namespace["_hg_src_to_obj"](source, str(SRC), descriptor["obj_dir"])
        for source in sources
    ]


def main() -> None:
    # Load the repository's audited DTK build helpers without asking setup.py to
    # build/install the much larger all-architecture Python package.
    os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
    original_setup = setuptools.setup
    setuptools.setup = lambda *args, **kwargs: None
    try:
        namespace = runpy.run_path(str(ROOT / "setup.py"), run_name="flash_attn_hg_build_helpers")
    finally:
        setuptools.setup = original_setup

    descriptor = namespace["compute_hg_build_descriptor"](
        SRC,
        BUILD,
        mode="1",
        extra_options_raw=(
            "-DHEADDIM_128_ONLY=ON "
            "-DGFX_VERSION=936 "
            "-Wl,-Bsymbolic"
        ),
    )
    adapt_flags_for_dcc_25_10(namespace, descriptor)
    select_h3_sources(namespace, descriptor)
    print(f"compiler={descriptor['compiler']}")
    print(f"sources={len(descriptor['sources'])}")
    print(f"output={descriptor['out_so']}")
    namespace["run_hg_ninja_build"](descriptor)


if __name__ == "__main__":
    main()
