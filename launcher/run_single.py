#!/usr/bin/env python3
"""Launch an isolated canonical single-GPU ComfyUI for quality baselines."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


gpu = os.environ["H3_SINGLE_GPU"]
state = Path(os.environ["H3_SINGLE_STATE"])
port = os.environ["H3_SINGLE_PORT"]
root = Path("/path/to/ComfyUI")

# Device visibility must be fixed before importing torch.
os.environ["HIP_VISIBLE_DEVICES"] = gpu
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

for name in ("input", "output", "temp", "user", "cache", "home", "custom_nodes"):
    (state / name).mkdir(parents=True, exist_ok=True)
(state / "user" / "default").mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(state / "home")
os.environ["XDG_CACHE_HOME"] = str(state / "cache")
os.environ["TMPDIR"] = str(state / "temp")

import torch

torch.cuda.set_device(0)
try:
    torch.ops.torchvision.nms
except (AttributeError, RuntimeError):
    torchvision_lib = torch.library.Library("torchvision", "DEF")
    torchvision_lib.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"single worker must see exactly one GPU, got {torch.cuda.device_count()}"
    )

sys.path.insert(0, str(root))
sys.argv = [
    str(root / "main.py"),
    "--listen", "127.0.0.1",
    "--port", port,
    "--base-directory", str(state),
    "--input-directory", str(state / "input"),
    "--output-directory", str(state / "output"),
    "--temp-directory", str(state / "temp"),
    "--user-directory", str(state / "user"),
    "--database-url", f"sqlite:///{state / 'user' / 'comfyui.db'}",
    "--extra-model-paths-config",
    "/path/to/runtime/extra_model_paths.yaml",
    "--disable-auto-launch",
    "--disable-metadata",
    "--disable-triton-backend",
    "--disable-xformers",
    "--gpu-only",
    "--disable-async-offload",
    "--log-stdout",
]

runpy.run_path(str(root / "main.py"), run_name="__main__")
