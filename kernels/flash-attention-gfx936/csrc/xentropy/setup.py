import os
import subprocess
from packaging.version import parse, Version
from typing import Optional
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
from get_version import get_version


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
cc_flag = ["--offload-arch=gfx906","--offload-arch=gfx926","--offload-arch=gfx928", "--offload-arch=gfx936"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="xentropy_cuda_lib",
        sources=[
            "interface.cpp",
            "xentropy_kernel.cu"
        ],
        extra_compile_args={'cxx': ['-O3'],'nvcc': ['-O3'] + cc_flag}
    )
)

def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

setup(
    name="xentropy_cuda_lib",
    version=get_version(ROCM_HOME),
    description="Cross-entropy loss",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    install_requires=[
         f"torch=={_get_pytorch_version()}",
    ],
)
