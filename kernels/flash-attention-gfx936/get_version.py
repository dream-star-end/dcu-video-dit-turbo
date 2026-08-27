import os, re
import ast
import subprocess
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.resolve()


def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None

def get_package_version():
    with open(Path(os.path.dirname(os.path.abspath(__file__))) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def _make_version_file(version, sha, abi, dtk, torch_version, branch):
    sha = "Unknown" if sha is None else sha
    torch_version = '.'.join(torch_version.split('.')[:2])
    # hcu_version = f"{version}+das1.1git{sha}.abi{abi}.dtk{dtk}.torch{torch_version}"
    hcu_version = f"{version}+das.opt{os.environ['FLASH_ATTN_OPT']}.dtk{dtk}"
    version_path = ROOT_DIR / "flash_attn" / "version.py"
    with open(version_path, "w") as f:
        f.write(f"version = '{version}'\n")
        f.write(f"git_hash = '{sha}'\n")
        f.write(f"git_branch = '{branch}'\n")
        f.write(f"abi = 'abi{abi}'\n")
        f.write(f"dtk = '{dtk}'\n")
        f.write(f"torch_version = '{torch_version}'\n")
        f.write(f"hcu_version = '{hcu_version}'\n")
    return hcu_version


def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"{os.environ['PYTORCH_VERSION']}"
    return torch.__version__

def get_version(ROCM_HOME):
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    if sha is not None :
        sha = sha[:7]
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])
    print("-- Git branch:", branch)
    print("-- Git SHA:", sha)
    print("-- Git tag:", tag)
    torch_version = _get_pytorch_version()
    print("-- PyTorch:", torch_version)
    version = get_package_version()
    print("-- Building version", version)
    abi = _run_cmd(["echo '#include <string>' | gcc -x c++ -E -dM - | fgrep _GLIBCXX_USE_CXX11_ABI | awk '{print $3}'"], shell=True)
    print("-- _GLIBCXX_USE_CXX11_ABI:", abi)
    dtk = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk = ''.join(dtk.replace(' ', '').replace('-', '').replace('V', '').split('.'))
    print("-- DTK:", dtk)

    return _make_version_file(version, sha, abi, dtk, torch_version, branch)
