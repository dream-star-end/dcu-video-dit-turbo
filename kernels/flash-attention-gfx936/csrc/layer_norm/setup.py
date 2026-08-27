import os
import subprocess
from packaging.version import parse, Version
from typing import Optional
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from get_version import get_version
this_dir = os.path.dirname(os.path.abspath(__file__))
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
cc_flag = ["--offload-arch=gfx906","--offload-arch=gfx926","--offload-arch=gfx928","--offload-arch=gfx936"]
# cc_flag = ["--offload-arch=gfx906"]
ext_modules = []

ext_modules.append(
    CUDAExtension(
        name="dropout_layer_norm",
        sources=[
            "ln_api.cu",
            "ln_fwd_256.cu",
            "ln_bwd_256.cu",
            "ln_fwd_512.cu",
            "ln_bwd_512.cu",
            "ln_fwd_768.cu",
            "ln_bwd_768.cu",
            "ln_fwd_1024.cu",
            "ln_bwd_1024.cu",
            "ln_fwd_1280.cu",
            "ln_bwd_1280.cu",
            "ln_fwd_1536.cu",
            "ln_bwd_1536.cu",
            "ln_fwd_2048.cu",
            "ln_bwd_2048.cu",
            "ln_fwd_2560.cu",
            "ln_bwd_2560.cu",
            "ln_fwd_3072.cu",
            "ln_bwd_3072.cu",
            "ln_fwd_4096.cu",
            "ln_bwd_4096.cu",
            "ln_fwd_5120.cu",
            "ln_bwd_5120.cu",
            "ln_fwd_6144.cu",
            "ln_bwd_6144.cu",
            "ln_fwd_7168.cu",
            "ln_bwd_7168.cu",
            "ln_fwd_8192.cu",
            "ln_bwd_8192.cu",
            "ln_parallel_fwd_256.cu",
            "ln_parallel_bwd_256.cu",
            "ln_parallel_fwd_512.cu",
            "ln_parallel_bwd_512.cu",
            "ln_parallel_fwd_768.cu",
            "ln_parallel_bwd_768.cu",
            "ln_parallel_fwd_1024.cu",
            "ln_parallel_bwd_1024.cu",
            "ln_parallel_fwd_1280.cu",
            "ln_parallel_bwd_1280.cu",
            "ln_parallel_fwd_1536.cu",
            "ln_parallel_bwd_1536.cu",
            "ln_parallel_fwd_2048.cu",
            "ln_parallel_bwd_2048.cu",
            "ln_parallel_fwd_2560.cu",
            "ln_parallel_bwd_2560.cu",
            "ln_parallel_fwd_3072.cu",
            "ln_parallel_bwd_3072.cu",
            "ln_parallel_fwd_4096.cu",
            "ln_parallel_bwd_4096.cu",
            "ln_parallel_fwd_5120.cu",
            "ln_parallel_bwd_5120.cu",
            "ln_parallel_fwd_6144.cu",
            "ln_parallel_bwd_6144.cu",
            "ln_parallel_fwd_7168.cu",
            "ln_parallel_bwd_7168.cu",
            "ln_parallel_fwd_8192.cu",
            "ln_parallel_bwd_8192.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3","-w"] ,
            "nvcc": [ "-O3","-w",'-U__HIP_NO_HALF_OPERATORS__','-U__HIP_NO_HALF_CONVERSIONS__'] + cc_flag
        },
        include_dirs=[this_dir],
    )
)

def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

setup(
    name="dropout_layer_norm",
    version=get_version(ROCM_HOME),
    description="Fused dropout + add + layer norm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    install_requires=[
         f"torch=={_get_pytorch_version()}",
    ],
)
