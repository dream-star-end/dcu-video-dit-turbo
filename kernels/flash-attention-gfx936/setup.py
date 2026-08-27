# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
from typing import Optional
from get_version import get_version
from setuptools import setup, find_packages
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py as setuptools_build_py
import subprocess
import shutil
import glob
import shlex

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
import torch.utils.cpp_extension as torch_cpp_extension
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)
os.environ['CXX'] = 'hipcc'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def cutlass_include_dirs():
    candidates = [
        Path(this_dir) / "csrc" / "cutlass" / "include",
    ]
    cutlass_home = os.getenv("CUTLASS_HOME")
    if cutlass_home:
        candidates.append(Path(cutlass_home) / "include")
    return [str(path) for path in candidates if path.exists()]


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    # nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    # return nvcc_extra_args + ["--threads", nvcc_threads]
    return nvcc_extra_args


cmdclass = {}
ext_modules = []


# --- HG libflash_attention.so: descriptor + Ninja (inlined; no extra module file) ---

_HG_EXPLICIT_SOURCES_BY_MODE = {
    # mode=1: forward attention kernels (plus shared HG entrypoint).
    "1": [
        "flash_api.cpp",
        "src/target/flash_fwd_hdim128_attn_mask_bf16.cpp",
        "src/target/flash_fwd_hdim128_attn_mask_fp16.cpp",
        "src/target/flash_fwd_hdim128_bf16.cpp",
        "src/target/flash_fwd_hdim128_fp16.cpp",
        "src/target/flash_fwd_hdim128_padding_mask_bf16.cpp",
        "src/target/flash_fwd_hdim128_padding_mask_fp16.cpp",
        "src/target/flash_fp8_fwd_hdim128_bf16.cpp",
        "src/target/flash_fp8_fwd_hdim128_fp16.cpp",
        "src/target/flash_fp8_fwd_hdim128_prefix_prefill_bf16.cpp",
        "src/target/flash_fp8_fwd_hdim128_prefix_prefill_fp16.cpp",
        "src/target/flash_fwd_hdim128_prefix_prefill_bf16.cpp",
        "src/target/flash_fwd_hdim128_prefix_prefill_fp16.cpp",
        "src/target/flash_fp8_fwd_hdim128_prefix_prefill_bf16.cpp",
        "src/target/flash_fp8_fwd_hdim128_prefix_prefill_fp16.cpp",
        "src/target/flash_fp8_fwd_hdimqk192_hdimv128_prefix_prefill_bf16.cpp",
        "src/target/flash_fp8_fwd_hdimqk192_hdimv128_prefix_prefill_fp16.cpp",
        "src/target/flash_fp8_fwd_hdim256_prefix_prefill_bf16.cpp",
        "src/target/flash_fp8_fwd_hdim256_prefix_prefill_fp16.cpp",
        "src/target/flash_fwd_hdim160_bf16.cpp",
        "src/target/flash_fwd_hdim160_fp16.cpp",
        "src/target/flash_fwd_hdim192_bf16.cpp",
        "src/target/flash_fwd_hdim192_fp16.cpp",
        "src/target/flash_fwd_hdim224_bf16.cpp",
        "src/target/flash_fwd_hdim224_fp16.cpp",
        "src/target/flash_fwd_hdim256_bf16.cpp",
        "src/target/flash_fwd_hdim256_fp16.cpp",
        "src/target/flash_fwd_hdim32_bf16.cpp",
        "src/target/flash_fwd_hdim32_fp16.cpp",
        "src/target/flash_fwd_hdim512_bf16.cpp",
        "src/target/flash_fwd_hdim512_fp16.cpp",
        "src/target/flash_fwd_hdim512_prefix_prefill_bf16.cpp",
        "src/target/flash_fwd_hdim512_prefix_prefill_fp16.cpp",
        "src/target/flash_fwd_hdim64_bf16.cpp",
        "src/target/flash_fwd_hdim64_fp16.cpp",
        "src/target/flash_fwd_hdim96_bf16.cpp",
        "src/target/flash_fwd_hdim96_fp16.cpp",
        "src/target/flash_fwd_hdimqk192_hdimv128_bf16.cpp",
        "src/target/flash_fwd_hdimqk192_hdimv128_fp16.cpp",
        "src/target/flash_fwd_prefix_prefill_mla_bf16.cpp",
        "src/target/flash_fwd_prefix_prefill_mla_fp16.cpp",
        "src/target/flash_int8_fwd_hdim128_prefix_prefill_bf16.cpp",
        "src/target/flash_int8_fwd_hdim128_prefix_prefill_fp16.cpp",
    ],
    # mode=2: backward kernels (plus shared HG entrypoint).
    "2": [
        "flash_api.cpp",
        "src/target/flash_bwd_hdim128_bf16.cpp",
        "src/target/flash_bwd_hdim128_fp16.cpp",
        "src/target/flash_bwd_hdim160_bf16.cpp",
        "src/target/flash_bwd_hdim160_fp16.cpp",
        "src/target/flash_bwd_hdim192_bf16.cpp",
        "src/target/flash_bwd_hdim192_fp16.cpp",
        "src/target/flash_bwd_hdim224_bf16.cpp",
        "src/target/flash_bwd_hdim224_fp16.cpp",
        "src/target/flash_bwd_hdim256_bf16.cpp",
        "src/target/flash_bwd_hdim256_fp16.cpp",
        "src/target/flash_bwd_hdim32_bf16.cpp",
        "src/target/flash_bwd_hdim32_fp16.cpp",
        "src/target/flash_bwd_hdim64_bf16.cpp",
        "src/target/flash_bwd_hdim64_fp16.cpp",
        "src/target/flash_bwd_hdim96_bf16.cpp",
        "src/target/flash_bwd_hdim96_fp16.cpp",
        "src/target/flash_bwd_hdimqk192_hdimv128_bf16.cpp",
        "src/target/flash_bwd_hdimqk192_hdimv128_fp16.cpp",
    ],
    # mode=3: split-kv / kv-cache kernels (plus shared HG entrypoint).
    "3": [
        "flash_api.cpp",
        "src/target/flash_fwd_split_hdim128_bf16.cpp",
        "src/target/flash_fwd_split_hdim128_fp16.cpp",
        "src/target/flash_fwd_split_hdim160_bf16.cpp",
        "src/target/flash_fwd_split_hdim160_fp16.cpp",
        "src/target/flash_fwd_split_hdim192_bf16.cpp",
        "src/target/flash_fwd_split_hdim192_fp16.cpp",
        "src/target/flash_fwd_split_hdim224_bf16.cpp",
        "src/target/flash_fwd_split_hdim224_fp16.cpp",
        "src/target/flash_fwd_split_hdim256_bf16.cpp",
        "src/target/flash_fwd_split_hdim256_fp16.cpp",
        "src/target/flash_fwd_split_hdim32_bf16.cpp",
        "src/target/flash_fwd_split_hdim32_fp16.cpp",
        "src/target/flash_fwd_split_hdim512_bf16.cpp",
        "src/target/flash_fwd_split_hdim512_fp16.cpp",
        "src/target/flash_fwd_split_hdim64_bf16.cpp",
        "src/target/flash_fwd_split_hdim64_fp16.cpp",
        "src/target/flash_fwd_split_hdim96_bf16.cpp",
        "src/target/flash_fwd_split_hdim96_fp16.cpp",
        "src/target/flash_fwd_split_hdimqk192_hdimv128_bf16.cpp",
        "src/target/flash_fwd_split_hdimqk192_hdimv128_fp16.cpp",
        "src/target/flash_fwd_split_hdimqk576_hdimv512_bf16.cpp",
        "src/target/flash_fwd_split_hdimqk576_hdimv512_fp16.cpp",
        "src/target/flash_int8_fwd_split_hdim128_bf16.cpp",
        "src/target/flash_int8_fwd_split_hdim128_fp16.cpp",
    ],
    # mode=4: permute kernels (plus shared HG entrypoint).
    "4": [
        "flash_api.cpp",
        "src/target/flash_fwd_permute_bhsd2bshd_hdim128.cpp",
        "src/target/flash_fwd_permute_bhsd2sbhd_hdim128.cpp",
        "src/target/flash_fwd_permute_bshd2bhsd_hdim128.cpp",
        "src/target/flash_fwd_permute_sbhd2bhsd_hdim128.cpp",
        "src/target/flash_varlen_fwd_permute_bhsd2bshd_hdim128.cpp",
        "src/target/flash_varlen_fwd_permute_bshd2bhsd_hdim128.cpp",
    ],
    # mode=5: MLA kernels (plus shared HG entrypoint).
    "5": [
        "flash_api.cpp",
        "src/target/flash_mla_hdimqk576_hdimv512_bf16.cpp",
        "src/target/flash_mla_hdimqk576_hdimv512_fp16.cpp",
        "src/target/flash_mla_hdimqk576_hdimv512_fp8.cpp",
    ],
}

# "all" is not stored separately: it is derived as ordered union(mode1..mode5).
_HG_EXPLICIT_MODE_ORDER = ("1", "2", "3", "4", "5")


def _resolve_hg_explicit_sources(src_dir: str, mode: str):
    mode_key = str(mode)
    if mode_key in _HG_EXPLICIT_MODE_ORDER:
        rel_sources = list(_HG_EXPLICIT_SOURCES_BY_MODE[mode_key])
    else:
        seen = set()
        rel_sources = []
        for _m in _HG_EXPLICIT_MODE_ORDER:
            for rel in _HG_EXPLICIT_SOURCES_BY_MODE[_m]:
                if rel not in seen:
                    seen.add(rel)
                    rel_sources.append(rel)
    abs_sources = [os.path.join(src_dir, rel) for rel in rel_sources]
    missing = [rel for rel, abs_path in zip(rel_sources, abs_sources) if not os.path.isfile(abs_path)]
    if missing:
        raise RuntimeError(
            "error: explicit HG source list is stale; missing files: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    return sorted(abs_sources)


def _maybe_clean_hg_build_dir(build_dir: str) -> None:
    """Full clean only when FLASH_ATTENTION_FORCE_CLEAN_HG=1 (default: incremental)."""
    if os.environ.get("FLASH_ATTENTION_FORCE_CLEAN_HG", "") != "1":
        return
    build_dir = os.path.abspath(build_dir)
    shutil.rmtree(os.path.join(build_dir, "objs"), ignore_errors=True)
    for pat in ("build_hg.ninja", ".ninja_deps", ".ninja_log", "libflash_attention.so", "libflash_attention.so.rsp"):
        p = os.path.join(build_dir, pat)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _hg_src_to_obj(src_path: str, src_root: str, obj_dir: str) -> str:
    rel = os.path.relpath(os.path.abspath(src_path), os.path.abspath(src_root))
    safe = rel.replace(os.sep, "_").replace("..", "_")
    return os.path.join(obj_dir, safe + ".o")


def _ninja_escape(s: str) -> str:
    return s.replace("$", "$$")


def _ninja_escape_path(s: str) -> str:
    return _ninja_escape(s).replace(" ", "$ ").replace(":", "$:")


def _ninja_shell_join(args) -> str:
    return " ".join(_ninja_escape(shlex.quote(str(x))) for x in args)


def _resolve_hg_compiler() -> str:
    candidates = [
        os.environ.get("FLASH_ATTN_HG_COMPILER"),
        "/opt/dtk/bin/aicc",
        "aicc",
    ]
    for compiler in candidates:
        if compiler and shutil.which(compiler):
            return compiler
    requested = [c for c in candidates if c]
    raise RuntimeError(
        "error: no usable HG aicc compiler found from: "
        + ", ".join(repr(c) for c in requested)
        + ". Set FLASH_ATTN_HG_COMPILER to the DTK aicc path."
    )


def _normalize_hg_gfx_archs(gfx_version: str):
    archs = []
    for item in str(gfx_version).replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        archs.append(item if item.startswith("gfx") else f"gfx{item}")
    return archs


def _hg_target_define_value(archs):
    return ",".join(arch[3:] if arch.startswith("gfx") else arch for arch in archs)


def _hg_arch_device_flags(archs):
    flags = []
    for arch in archs:
        if arch in ("gfx936", "gfx938"):
            flags.extend([f"-Xarch_{arch}", "-mllvm=-support-768-vgprs=true"])
        elif arch in ("gfx928", "gfx92a", "gfx946"):
            flags.extend([f"-Xarch_{arch}", "-mllvm=-support-512-vgprs=true"])
            flags.extend([f"-Xarch_{arch}", "-mllvm=-co-issue-vgpr-size=256"])
    return flags



# Keep per-arch -Xarch pairs intact until PyTorch writes build.ninja.
_MAIN_EXT_ARCH_FLAG_EXPANSIONS = {
    "__FLASH_ATTN_XARCH_938_SUPPORT_768_VGPRS__": [
        "-Xarch_gfx938",
        "-mllvm=-support-768-vgprs=true",
    ],
    "__FLASH_ATTN_XARCH_936_SUPPORT_768_VGPRS__": [
        "-Xarch_gfx936",
        "-mllvm=-support-768-vgprs=true",
    ],
    "__FLASH_ATTN_XARCH_928_SUPPORT_512_VGPRS__": [
        "-Xarch_gfx928",
        "-mllvm=-support-512-vgprs=true",
    ],
    "__FLASH_ATTN_XARCH_928_CO_ISSUE_256__": [
        "-Xarch_gfx928",
        "-mllvm=-co-issue-vgpr-size=256",
    ],
    "__FLASH_ATTN_XARCH_92A_SUPPORT_512_VGPRS__": [
        "-Xarch_gfx92a",
        "-mllvm=-support-512-vgprs=true",
    ],
    "__FLASH_ATTN_XARCH_92A_CO_ISSUE_256__": [
        "-Xarch_gfx92a",
        "-mllvm=-co-issue-vgpr-size=256",
    ],
}


def _main_ext_arch_device_flag_placeholders(archs):
    flags = []
    for arch in archs:
        suffix = arch[3:].upper()
        if arch in ("gfx936", "gfx938"):
            flags.append(f"__FLASH_ATTN_XARCH_{suffix}_SUPPORT_768_VGPRS__")
        elif arch in ("gfx928", "gfx92a"):
            flags.append(f"__FLASH_ATTN_XARCH_{suffix}_SUPPORT_512_VGPRS__")
            flags.append(f"__FLASH_ATTN_XARCH_{suffix}_CO_ISSUE_256__")
    return flags


def _expand_main_ext_arch_flag_placeholders(flags):
    if flags is None:
        return None
    expanded = []
    for flag in flags:
        replacement = _MAIN_EXT_ARCH_FLAG_EXPANSIONS.get(flag)
        if replacement is None:
            expanded.append(flag)
        else:
            expanded.extend(replacement)
    return expanded


def _patch_torch_write_ninja_file_for_main_ext_arch_flags():
    original = torch_cpp_extension._write_ninja_file
    if getattr(original, "_flash_attn_arch_flags_patched", False):
        return

    def patched_write_ninja_file(
        path,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        sycl_cflags,
        sycl_post_cflags,
        sycl_dlink_post_cflags,
        sources,
        objects,
        ldflags,
        library_target,
        with_cuda,
        with_sycl,
    ):
        cuda_post_cflags = _expand_main_ext_arch_flag_placeholders(cuda_post_cflags)
        cuda_dlink_post_cflags = _expand_main_ext_arch_flag_placeholders(cuda_dlink_post_cflags)
        return original(
            path,
            cflags,
            post_cflags,
            cuda_cflags,
            cuda_post_cflags,
            cuda_dlink_post_cflags,
            sycl_cflags,
            sycl_post_cflags,
            sycl_dlink_post_cflags,
            sources,
            objects,
            ldflags,
            library_target,
            with_cuda,
            with_sycl,
        )

    patched_write_ninja_file._flash_attn_arch_flags_patched = True
    torch_cpp_extension._write_ninja_file = patched_write_ninja_file


def _is_rocm_5_7() -> bool:
    version_file = "/opt/rocm/.info/version"
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.read(100).startswith("5.7.0")
    except OSError:
        return False


def compute_hg_build_descriptor(
    src_dir,
    build_dir,
    mode="all",
    extra_options_raw="-DGFX_VERSION=938;936;92a -Wl,-Bsymbolic",
):
    """Collect HG sources and flags for Ninja (no compile). Default: mode=all, gfx938/gfx936/gfx92a."""
    import sysconfig as _sysconfig

    src_dir = os.path.abspath(str(src_dir))
    build_dir = os.path.abspath(str(build_dir))
    obj_dir = os.path.join(build_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)

    BUILD_FA_FWD = BUILD_FA_BWD = BUILD_FA_KVCACHE = False
    BUILD_FA_PERMUTE = BUILD_FLASHMLA = False
    BUILD_C_INTERFACE = False
    BUILD_ASM = True
    FA_DEBUG = True
    FA_DEBUG_SUM_MAX = False
    HEADDIM_128_ONLY = False
    HEADDIM_192_128_ONLY = False
    FA_KERNEL_TIMER = False
    PA_PAGE_BLOCK_SIZE = False
    MLA_PAGE_BLOCK_SIZE = False

    if mode == "1":
        BUILD_FA_FWD = True
    elif mode == "2":
        BUILD_FA_BWD = True
    elif mode == "3":
        BUILD_FA_KVCACHE = True
    elif mode == "4":
        BUILD_FA_PERMUTE = True
    elif mode == "5":
        BUILD_FLASHMLA = True
    else:
        BUILD_FA_FWD = BUILD_FA_BWD = BUILD_FA_KVCACHE = True
        BUILD_FA_PERMUTE = BUILD_FLASHMLA = True

    EXTRA_HIP_FLAGS = []
    EXTRA_LINK_FLAGS = []
    GFX_VERSION = None

    for _tok in extra_options_raw.split():
        if _tok in ("ninja", "-G", "Ninja"):
            pass
        elif _tok == "-DBUILD_C_INTERFACE=ON":
            BUILD_C_INTERFACE = True
        elif _tok == "-DBUILD_C_INTERFACE=OFF":
            BUILD_C_INTERFACE = False
        elif _tok == "-DBUILD_ASM=ON":
            BUILD_ASM = True
        elif _tok == "-DBUILD_ASM=OFF":
            BUILD_ASM = False
        elif _tok == "-DFA_DEBUG=ON":
            FA_DEBUG = True
        elif _tok == "-DFA_DEBUG=OFF":
            FA_DEBUG = False
        elif _tok == "-DFA_DEBUG_SUM_MAX=ON":
            FA_DEBUG_SUM_MAX = True
        elif _tok == "-DFA_DEBUG_SUM_MAX=OFF":
            FA_DEBUG_SUM_MAX = False
        elif _tok == "-DHEADDIM_128_ONLY=ON":
            HEADDIM_128_ONLY = True
        elif _tok == "-DHEADDIM_128_ONLY=OFF":
            HEADDIM_128_ONLY = False
        elif _tok == "-DHEADDIM_192_128_ONLY=ON":
            HEADDIM_192_128_ONLY = True
        elif _tok == "-DHEADDIM_192_128_ONLY=OFF":
            HEADDIM_192_128_ONLY = False
        elif _tok == "-DFA_KERNEL_TIMER=ON":
            FA_KERNEL_TIMER = True
        elif _tok == "-DFA_KERNEL_TIMER=OFF":
            FA_KERNEL_TIMER = False
        elif _tok == "-DPA_PAGE_BLOCK_SIZE=ON":
            PA_PAGE_BLOCK_SIZE = True
        elif _tok == "-DPA_PAGE_BLOCK_SIZE=OFF":
            PA_PAGE_BLOCK_SIZE = False
        elif _tok == "-DMLA_PAGE_BLOCK_SIZE=ON":
            MLA_PAGE_BLOCK_SIZE = True
        elif _tok == "-DMLA_PAGE_BLOCK_SIZE=OFF":
            MLA_PAGE_BLOCK_SIZE = False
        elif _tok.startswith("-DGFX_VERSION="):
            GFX_VERSION = _tok[len("-DGFX_VERSION=") :]
        elif _tok.startswith(("-j", "--jobs=")) or _tok == "VERBOSE=1":
            pass
        elif _tok.startswith(("-Wl,", "-l", "-L")):
            EXTRA_LINK_FLAGS.append(_tok)
        else:
            EXTRA_HIP_FLAGS.append(_tok)

    if GFX_VERSION is None:
        GFX_VERSION = "938;936;92a"

    ROCM_PATH = os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME", "/opt/rocm"))
    HG_COMPILER = _resolve_hg_compiler()

    if not os.path.isdir(os.path.join(ROCM_PATH, "include")):
        raise RuntimeError(
            f"error: {ROCM_PATH}/include not found. "
            "Please set ROCM_PATH/ROCM_HOME to the DTK toolchain root."
        )

    TORCH_INCLUDE_FLAGS = []
    TORCH_LINK_FLAGS = []
    PYTHON_INC = ""

    if not BUILD_C_INTERFACE:
        PYTHON_INC = _sysconfig.get_paths()["include"]
        _py_libdir = _sysconfig.get_config_var("LIBDIR") or ""
        _ldlib = (
            _sysconfig.get_config_var("LDLIBRARY")
            or f"libpython{sys.version_info.major}.{sys.version_info.minor}.so"
        )
        _py_libname = os.path.basename(_ldlib)
        if _py_libname.startswith("lib"):
            _py_libname = _py_libname[3:]
        _py_libname = re.sub(r"\.so.*$", "", _py_libname)

        from torch.utils import cpp_extension as _cpp_ext

        TORCH_INCLUDE_FLAGS = [f"-I{_inc}" for _inc in _cpp_ext.include_paths()]
        _torch_libdir = str(Path(torch.__file__).resolve().parent / "lib")
        TORCH_LINK_FLAGS = [
            f"-L{_torch_libdir}",
            f"-L{_py_libdir}",
            "-Wl,--no-as-needed",
            "-ltorch_hip",
            "-ltorch_cpu",
            "-ltorch_python",
            f"-l{_py_libname}",
            "-lc10",
        ]

    HG_ARCHS = _normalize_hg_gfx_archs(GFX_VERSION)
    _gfx_comma = _hg_target_define_value(HG_ARCHS)
    DEFINES = [
        f"-DTARGET={_gfx_comma}",
        "-D__HIP_PLATFORM_AMD__=1",
        "-DUSE_ROCM=1",
        "-DCUDA_HAS_FP16=1",
        "-DNDEBUGING",
    ]
    if BUILD_C_INTERFACE:
        DEFINES.append("-DBUILD_C_INTERFACE")
    else:
        DEFINES += [
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            '-DPYBIND11_COMPILER_TYPE="_gcc"',
            '-DPYBIND11_STDLIB="_libstdcpp"',
            '-DPYBIND11_BUILD_ABI="_cxxabi1016"',
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ]
    if BUILD_ASM:
        DEFINES.append("-DBUILD_ASM")
    if BUILD_FA_FWD:
        DEFINES.append("-DBUILD_FA_FWD")
    if BUILD_FA_BWD:
        DEFINES.append("-DBUILD_FA_BWD")
    if BUILD_FA_KVCACHE:
        DEFINES.append("-DBUILD_FA_KVCACHE")
    if BUILD_FA_PERMUTE:
        DEFINES.append("-DBUILD_FA_PERMUTE")
    if BUILD_FLASHMLA:
        DEFINES.append("-DBUILD_FLASHMLA")
    if FA_DEBUG:
        DEFINES.append("-DFA_DEBUG")
    if FA_DEBUG_SUM_MAX:
        DEFINES.append("-DFA_DEBUG_SUM_MAX")
    if HEADDIM_128_ONLY:
        DEFINES.append("-DHEADDIM_128_ONLY")
    if HEADDIM_192_128_ONLY:
        DEFINES.append("-DHEADDIM_192_128_ONLY")
    if FA_KERNEL_TIMER:
        DEFINES.append("-DFA_KERNEL_TIMER")
    if PA_PAGE_BLOCK_SIZE:
        DEFINES.append("-DPA_PAGE_BLOCK_SIZE")
    if MLA_PAGE_BLOCK_SIZE:
        DEFINES.append("-DMLA_PAGE_BLOCK_SIZE")
    if _is_rocm_5_7():
        DEFINES.append("-DROCM_5_7")

    OFFLOAD_FLAGS = [f"--offload-arch={_g}" for _g in HG_ARCHS]

    INCLUDE_FLAGS = [
        f"-I{ROCM_PATH}/include",
        f"-I{src_dir}/include",
        f"-I{src_dir}/src",
    ]
    if not BUILD_C_INTERFACE:
        INCLUDE_FLAGS.append(f"-I{PYTHON_INC}")
        INCLUDE_FLAGS += TORCH_INCLUDE_FLAGS

    COMMON_FLAGS = [
        "-fPIC",
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-fno-finite-math-only",
        "-fno-gpu-rdc",
    ]
    DTK_DEVICE_FLAGS = [
        "-mllvm",
        "-disable-machine-sink",
        "-mllvm",
        "-disable-code-sink",
        "-mcode-object-version=5",
    ]
    if not _is_rocm_5_7():
        DTK_DEVICE_FLAGS += [
            "-mllvm",
            "-amdgpu-enable-rewrite-partial-reg-uses=false",
            "-mllvm",
            "-allow-gvn-convergent-call=true",
            "-mllvm",
            "-disallow-uniform-vmed3-combine=true",
            "-mllvm",
            "-hcu-pre-emit-load-store-opt=false",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
        ]
    DTK_DEVICE_FLAGS += _hg_arch_device_flags(HG_ARCHS)
    if os.environ.get("FLASH_ATTN_HG_SAVE_TEMPS", "") == "1":
        DTK_DEVICE_FLAGS.append("--save-temps")

    _env_extra = os.environ.get("FLASH_ATTN_DTK_EXTRA_FLAGS", "")
    if _env_extra:
        EXTRA_HIP_FLAGS += _env_extra.split()

    # Default path: use explicit in-code source lists so build targets are deterministic and auditable.
    # Fallback to glob only for special head-dim override modes to preserve legacy behavior.
    if not HEADDIM_128_ONLY and not HEADDIM_192_128_ONLY:
        _all_sources = _resolve_hg_explicit_sources(src_dir, mode)
    else:
        warnings.warn(
            "HEADDIM_*_ONLY is enabled; falling back to pattern-based source selection for HG build.",
            RuntimeWarning,
        )
        _tgt = os.path.join(src_dir, "src", "target")
        target_sources = []

        if BUILD_FLASHMLA:
            target_sources += glob.glob(os.path.join(_tgt, "flash_mla*.cpp"))

        if BUILD_FA_FWD:
            if HEADDIM_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*fwd_hdim128_*.cpp"))
                target_sources += glob.glob(os.path.join(_tgt, "flash_fwd_prefix_prefill_mla*.cpp"))
            elif HEADDIM_192_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*fwd_hdimqk192_hdimv128_*.cpp"))
            else:
                target_sources += glob.glob(os.path.join(_tgt, "*fwd_hdim*.cpp"))
                target_sources += glob.glob(os.path.join(_tgt, "flash_fwd_prefix_prefill_mla*.cpp"))

        if BUILD_FA_BWD:
            if HEADDIM_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*bwd_hdim128_*.cpp"))
            elif HEADDIM_192_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*bwd_hdimqk192_hdimv128_*.cpp"))
            else:
                target_sources += glob.glob(os.path.join(_tgt, "*bwd*.cpp"))

        if BUILD_FA_KVCACHE:
            if HEADDIM_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*split_hdim128_*.cpp"))
            elif HEADDIM_192_128_ONLY:
                target_sources += glob.glob(os.path.join(_tgt, "*split_hdimqk*_hdimv*_*.cpp"))
            else:
                target_sources += glob.glob(os.path.join(_tgt, "*split_hdim*.cpp"))

        if BUILD_FA_PERMUTE:
            target_sources += glob.glob(os.path.join(_tgt, "*fwd_permute*.cpp"))

        cpp_sources = glob.glob(os.path.join(src_dir, "*.cpp"))

        if not target_sources:
            raise RuntimeError("error: no target source files selected")
        if not cpp_sources:
            raise RuntimeError(f"error: no top-level cpp source files found under {src_dir}")

        _all_sources = sorted(
            [s for s in target_sources if os.path.isfile(s)]
            + [s for s in cpp_sources if os.path.isfile(s)]
        )

    compile_flag_list = (
        COMMON_FLAGS + DTK_DEVICE_FLAGS + OFFLOAD_FLAGS + INCLUDE_FLAGS + DEFINES + EXTRA_HIP_FLAGS
    )
    compile_flags = _ninja_shell_join(compile_flag_list)

    link_flag_list = COMMON_FLAGS + OFFLOAD_FLAGS + EXTRA_HIP_FLAGS + TORCH_LINK_FLAGS + EXTRA_LINK_FLAGS
    link_flags = _ninja_shell_join(link_flag_list)

    objects = [_hg_src_to_obj(s, src_dir, obj_dir) for s in _all_sources]
    out_so = os.path.join(build_dir, "libflash_attention.so")
    ninja_path = os.path.join(build_dir, "build_hg.ninja")

    return {
        "src_dir": src_dir,
        "build_dir": build_dir,
        "obj_dir": obj_dir,
        "sources": _all_sources,
        "objects": objects,
        "compiler": HG_COMPILER,
        "compile_flags": compile_flags,
        "link_flags": link_flags,
        "out_so": out_so,
        "ninja_path": ninja_path,
    }


def run_hg_ninja_build(descriptor: dict) -> None:
    """Write build_hg.ninja and run ninja (parallel via MAX_JOBS)."""
    build_dir = descriptor["build_dir"]
    ninja_file = descriptor["ninja_path"]
    compiler = _ninja_shell_join([descriptor["compiler"]])
    out_so_ninja = _ninja_escape_path(descriptor["out_so"])
    lines = [
        "ninja_required_version = 1.3",
        "",
        "rule hg_compile",
        f"  command = {compiler} -c $in -o $out $FLAGS",
        "  description = HG compile $in",
        "",
        "rule hg_link",
        f"  command = {compiler} -shared -o $out @$out.rsp $LINK_FLAGS",
        "  rspfile = $out.rsp",
        "  rspfile_content = $in",
        "  description = HG link $out",
        "",
        f"FLAGS = {descriptor['compile_flags']}",
        f"LINK_FLAGS = {descriptor['link_flags']}",
        "",
    ]
    for src, obj in zip(descriptor["sources"], descriptor["objects"]):
        lines.append(f"build {_ninja_escape_path(obj)}: hg_compile {_ninja_escape_path(src)}")
    obj_list = " ".join(_ninja_escape_path(obj) for obj in descriptor["objects"])
    lines.append(f"build {out_so_ninja}: hg_link {obj_list}")
    lines.append("")

    os.makedirs(build_dir, exist_ok=True)
    with open(ninja_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ninja_bin = shutil.which("ninja")
    if not ninja_bin:
        raise RuntimeError("ninja not found in PATH (required for HG parallel build)")
    max_jobs = os.environ.get("MAX_JOBS", str(max(1, (os.cpu_count() or 2) // 2)))
    subprocess.check_call(
        [ninja_bin, "-f", ninja_file, "-C", build_dir, "-j", max_jobs],
    )


def build_hg(src_dir, build_dir, mode="all", extra_options_raw="-DGFX_VERSION=938;936;92a -Wl,-Bsymbolic"):
    """Backward-compatible: clean (if env) + descriptor + ninja."""
    _maybe_clean_hg_build_dir(build_dir)
    desc = compute_hg_build_descriptor(src_dir, build_dir, mode=mode, extra_options_raw=extra_options_raw)
    run_hg_ninja_build(desc)


BUILD_HG = os.getenv("FLASH_BUILD_HG", "1") != "0"
HG_SRC_DIR = os.path.join(this_dir, "csrc", "flash_attn_hipc")
HG_BUILD_DIR = os.path.join(this_dir, "build", "flash_attn_hipc")
HG_SO_BUILD = os.path.join(HG_BUILD_DIR, "libflash_attention.so")
HG_SO_PKG = os.path.join(this_dir, "flash_attn", "lib", "libflash_attention.so")
HG_LIB_DIR = os.path.dirname(HG_SO_PKG)
os.environ['PYTORCH_NVCC'] = 'aicc'

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
# subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # BFloat16 rounding mode configuration (0: round_half_ulp_truncate (default), 1: round_toward_zero, 2: round_to_nearest)
    bf16_type = os.getenv("FLASH_ATTENTION_BF16_TYPE", "0")
    if bf16_type != "0":
        bf16_mode_names = {"1": "round_toward_zero", "2": "round_to_nearest"}
        print(f"Using BFloat16 rounding mode: {bf16_mode_names.get(bf16_type, 'unknown')}")

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    # if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    #     generator_flag = ["-DOLD_GENERATOR_PATH"]

    # check_if_cuda_home_none("flash_attn")
    # # Check, if CUDA11 is installed for compute capability 8.0
    # cc_flag = []
    # if CUDA_HOME is not None:
    #     _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    #     if bare_metal_version < Version("11.6"):
    #         raise RuntimeError(
    #             "FlashAttention is only supported on CUDA 11.6 and above.  "
    #             "Note: make sure nvcc has a supported version by running nvcc -V."
    #         )
    # # cc_flag.append("-gencode")
    # # cc_flag.append("arch=compute_75,code=sm_75")
    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_80,code=sm_80")
    # if CUDA_HOME is not None:
    #     if bare_metal_version >= Version("11.8"):
    #         cc_flag.append("-gencode")
    #         cc_flag.append("arch=compute_90,code=sm_90")


    # --- HG: libflash_attention.so is built in NinjaBuildExtension.run (ninja, mode=all, gfx938/gfx936/gfx92a) ---
    # HAS_HG_DISPATCH / -lflash_attention are applied there if the .so exists.
    hg_compile_defs = []
    hg_link_args = []
    main_ext_archs = _normalize_hg_gfx_archs("gfx938,gfx936,gfx928,gfx92a")
    main_ext_arch_flags = _main_ext_arch_device_flag_placeholders(main_ext_archs)
    _patch_torch_write_ninja_file_for_main_ext_arch_flags()
    main_ext_arch_flags_for_log = _expand_main_ext_arch_flag_placeholders(main_ext_arch_flags)
    print(
        "=== FlashAttention main extension archs: "
        f"{','.join(main_ext_archs)}; arch flags: {' '.join(main_ext_arch_flags_for_log) or '<none>'} ==="
    )
    aicc_flags = [
        "-mcode-object-version=5",
        "-mllvm=-disable-machine-sink",
        "-mllvm=-disable-code-sink",
        "-mllvm=-amdgpu-enable-rewrite-partial-reg-uses=false",
        "-mllvm=-allow-gvn-convergent-call=true",
        "-mllvm=-disallow-uniform-vmed3-combine=true",
        "-mllvm=-hcu-pre-emit-load-store-opt=false",
        "-mllvm=-amdgpu-early-inline-all=true",
        "-mllvm=-amdgpu-function-calls=false",
        "-fno-finite-math-only",
        "--gpu-max-threads-per-block=256",
        "-mllvm=-unroll-threshold=10000"
    ]
    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn_cutlass/flash_api_sparse.cpp",
                "csrc/flash_attn_cutlass/flash_api_attnmask.cpp",
                "csrc/flash_attn_cutlass/flash_api.cpp",
                "csrc/flash_attn_cutlass/src/flash_varlen_fwd_tiny_hdim64.cu",
                "csrc/flash_attn_cutlass/src/paged_attention.cu",
                "csrc/flash_attn_cutlass/src/paged_attention_938.cu",
                "csrc/flash_attn_cutlass/src/paged_attention_bhsd.cu",
                "csrc/flash_attn_cutlass/src/paged_attention_938_bhsd.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_padding_mask_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim32_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim32_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_attnmask_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim160_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim160_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim224_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim224_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim512_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim512_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim160_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim160_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim192_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim224_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim224_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim512_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_hdim512_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_sparse_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_blasst_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_blasst_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_blasst_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_blasst_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim160_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim160_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim224_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim224_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim512_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim512_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_attnmask_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim160_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim160_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim224_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim224_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim512_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_bwd_hdim512_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_outfp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_outfp16_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_q_bf16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_q_fp16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_outfp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_outfp16_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_q_bf16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_q_fp16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim160_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim160_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_outfp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_outfp16_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim224_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim224_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_outfp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_outfp16_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_q_bf16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_q_fp16_kv_e5m2_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_outfp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_fp8_outfp16_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_q_bf16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim64_q_fp16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_outfp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_fp8_outfp16_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_q_bf16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim128_q_fp16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_outfp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim192_fp8_outfp16_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim224_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim224_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_outfp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_fp8_outfp16_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_q_bf16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_split_hdim256_q_fp16_kv_e5m2_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim128_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim128_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim256_fp16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_fwd_unified_hdim256_bf16_sm80.cu",
                "csrc/flash_attn_cutlass/src/flash_sparse_util.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-w", "-std=c++17",
                        "-DDCU_ASM",
                        f"-DFLASH_ATTENTION_BF16_TYPE={bf16_type}",] + generator_flag + hg_compile_defs,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-w",
                        # "-g",
                        # "-ggdb",
                        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                        "-ffast-math",
                        "-mno-fma",
                        "-std=c++17",
                        "-DDCU_ASM",
                        # "-mllvm -not-combine-fma=true",
                        # "-mllvm -slp-phi-tree-bb-max-size=10000",
                        # "-mllvm -allow-cse-cross-bb-convergent-call=true",
                        # "-mllvm -full-vectorize-slp=true",
                        f"-DFLASH_ATTENTION_BF16_TYPE={bf16_type}",
                        # "-DHG_ROCM",
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DBWDTRANS",
                        "-DBWDSEPARATE",
                        # "-DFLASHATTENTION_DISABLE_SPLITKV",
                        "-DBF162FLOAT_USE_ASM",
                        "--offload-arch=gfx938,gfx936,gfx928",
                        # "-DGEMM1_AMATRIX_WITH_SMEM",
                        # "-U__CUDA_NO_HALF_OPERATORS__",
                        # "-U__CUDA_NO_HALF_CONVERSIONS__",
                        # "-U__CUDA_NO_HALF2_OPERATORS__",
                        # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        # "--expt-relaxed-constexpr",
                        # "--expt-extended-lambda",
                        # "--use_fast_math",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DFLASHATTENTION_DISABLE_DROPOUT",
                        # "-DFLASHATTENTION_DISABLE_ALIBI",
                        "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        # "-DFLASHATTENTION_DISABLE_LOCAL",
                        # "-Rpass-analysis=kernel-resource-usage",
                        # "--gpu-max-threads-per-block=1024",
                        "-mllvm",
                        "-enable-num-vgprs-512=true",
                        "-Rpass-analysis=kernel-resource-usage",
                        "--save-temps",
                        "-ftemplate-backtrace-limit=0",
                        # "--offload-compress",
                        # "--offload-compress-format=zlib",
                    ]
                    + generator_flag
                    + hg_compile_defs
                    + main_ext_arch_flags
                    + aicc_flags
                    # + cc_flag
                ),
            },
            extra_link_args=hg_link_args,
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn_cutlass",
                Path(this_dir) / "csrc" / "flash_attn_cutlass" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.hip)
    torch_version_raw = parse(torch.__version__)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.3
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        return super().run()
        # if FORCE_BUILD:
        #     return super().run()

        # wheel_url, wheel_filename = get_wheel_url()
        # print("Guessing wheel URL: ", wheel_url)
        # try:
        #     urllib.request.urlretrieve(wheel_url, wheel_filename)

        #     # Make the archive
        #     # Lifted from the root wheel processing command
        #     # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
        #     if not os.path.exists(self.dist_dir):
        #         os.makedirs(self.dist_dir)

        #     impl_tag, abi_tag, plat_tag = self.get_tag()
        #     archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

        #     wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        #     print("Raw wheel path", wheel_path)
        #     os.rename(wheel_filename, wheel_path)
        # except (urllib.error.HTTPError, urllib.error.URLError):
        #     print("Precompiled wheel not found. Building from source...")
        #     # If the wheel could not be downloaded, build from source
        #     super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

    def run(self):
        # HG first so flash_attn/lib/libflash_attention.so exists before build_py copies package_data.
        # Dispatch is enabled only when HG is explicitly requested and this run produced a valid .so.
        use_hg = False
        if not SKIP_CUDA_BUILD and BUILD_HG:
            # Remove stale packaged HG .so before making a fresh decision for this run.
            if os.path.isfile(HG_SO_PKG):
                try:
                    os.remove(HG_SO_PKG)
                except OSError:
                    pass
            hg_reuse_so = os.environ.get("FLASH_ATTN_HG_REUSE_SO", "")
            if hg_reuse_so:
                hg_reuse_so = os.path.abspath(hg_reuse_so)
                if not os.path.isfile(hg_reuse_so):
                    raise RuntimeError(f"Error: FLASH_ATTN_HG_REUSE_SO does not exist ({hg_reuse_so})")
                os.makedirs(os.path.dirname(HG_SO_PKG), exist_ok=True)
                shutil.copy2(hg_reuse_so, HG_SO_PKG)
                use_hg = True
                print(f"=== Reused HG .so {hg_reuse_so} -> {HG_SO_PKG} ===")
            elif os.path.isdir(HG_SRC_DIR):
                os.makedirs(HG_BUILD_DIR, exist_ok=True)
                _maybe_clean_hg_build_dir(HG_BUILD_DIR)
                try:
                    desc = compute_hg_build_descriptor(
                        HG_SRC_DIR,
                        HG_BUILD_DIR,
                        mode="all",
                        extra_options_raw="-DGFX_VERSION=938;936 -Wl,-Bsymbolic",
                    )
                    print(
                        "=== Building HG libflash_attention.so "
                        f"(mode=all, gfx938/gfx936, ninja, compiler={desc['compiler']}) ==="
                    )
                    run_hg_ninja_build(desc)
                    if os.path.isfile(HG_SO_BUILD):
                        os.makedirs(os.path.dirname(HG_SO_PKG), exist_ok=True)
                        shutil.copy2(HG_SO_BUILD, HG_SO_PKG)
                        use_hg = True
                        print(f"=== Copied HG .so -> {HG_SO_PKG} ===")
                    else:
                        raise RuntimeError("Error: HG build completed but output .so is missing")
                except Exception as e:
                    raise RuntimeError(f"Error: HG build failed ({e})")
            else:
                raise RuntimeError(f"Error: HG source directory not found ({HG_SRC_DIR})")
        else:
            # FLASH_BUILD_HG=0 should deterministically disable dispatch even if stale artifacts exist.
            if os.path.isfile(HG_SO_PKG):
                try:
                    os.remove(HG_SO_PKG)
                    print("=== Removed stale packaged HG .so because FLASH_BUILD_HG=0 or CUDA build is skipped ===")
                except OSError:
                    pass
        if use_hg:
            for ext in self.extensions:
                if ext.name == "flash_attn_2_cuda":
                    cxx_args = ext.extra_compile_args.setdefault("cxx", [])
                    if "-DHAS_HG_DISPATCH" not in cxx_args:
                        cxx_args.append("-DHAS_HG_DISPATCH")
                    nvcc_args = ext.extra_compile_args.setdefault("nvcc", [])
                    if "-DHAS_HG_DISPATCH" not in nvcc_args:
                        nvcc_args.append("-DHAS_HG_DISPATCH")

                    link_args = list(ext.extra_link_args or [])
                    for _arg in (f"-L{HG_LIB_DIR}", "-lflash_attention", "-Wl,-rpath,$ORIGIN/flash_attn/lib"):
                        if _arg not in link_args:
                            link_args.append(_arg)
                    ext.extra_link_args = link_args
            print("=== HG dispatch ENABLED ===")
        else:
            print("=== HG dispatch DISABLED (no .so found) ===")
        super().run()


class FlashAttentionBuild(setuptools_build):
    """Run build_ext before build_py so flash_attn/lib/libflash_attention.so exists for package_data."""

    def get_sub_commands(self):
        cmds = []
        if self.distribution.has_ext_modules():
            cmds.append("build_ext")
        if self.distribution.has_pure_modules():
            cmds.append("build_py")
        if self.has_c_libraries():
            cmds.append("build_clib")
        if self.has_scripts():
            cmds.append("build_scripts")
        return cmds


class FlashHgOptionalBuildPy(setuptools_build_py):
    """If HG build failed, drop *.so package_data patterns with no matches (optional wheel content)."""

    def run(self):
        pd = getattr(self.distribution, "package_data", None) or {}
        if "flash_attn" in pd:
            pkg_root = os.path.join(this_dir, "flash_attn")
            newp = []
            for pat in pd["flash_attn"]:
                matches = glob.glob(os.path.join(pkg_root, pat))
                has_glob = any(ch in pat for ch in "*?[]")
                if matches:
                    newp.append(pat)
                elif ".so" in pat or has_glob:
                    print(f"=== Skipping optional package_data flash_attn/{pat} (no matching files) ===")
                else:
                    newp.append(pat)
            self.distribution.package_data = dict(pd)
            self.distribution.package_data["flash_attn"] = newp
        super().run()


def _find_rocm_home() -> Optional[str]:
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        try:
            pipe_hipcc = subprocess.Popen(
                ["which hipcc | xargs readlink -f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            hipcc, _ = pipe_hipcc.communicate()
            rocm_home = os.path.dirname(os.path.dirname(hipcc.decode(*()).rstrip('\r\n')))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        except Exception:
            rocm_home = '/opt/rocm'
            if not os.path.exists(rocm_home):
                rocm_home = None
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'")
    return rocm_home


ROCM_HOME = _find_rocm_home()
pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')


setup(
    name=PACKAGE_NAME,
    # version=get_package_version(),
    version=get_version(ROCM_HOME),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    package_data=(
        {"flash_attn": ["lib/*.so"]}
        if (BUILD_HG and not SKIP_CUDA_BUILD)
        else {}
    ),
    ext_modules=ext_modules,
    cmdclass=(
        {
            "build": FlashAttentionBuild,
            "build_ext": NinjaBuildExtension,
            "build_py": FlashHgOptionalBuildPy,
            "bdist_wheel": CachedWheelsCommand,
        }
        if ext_modules
        else {
            "bdist_wheel": CachedWheelsCommand,
        }
    ),
    python_requires=">=3.8",
    install_requires=[
        pytorch_dep,
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
