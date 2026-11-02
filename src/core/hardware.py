"""Compute-device detection (local CPU vs. Colab / cloud CUDA GPU)."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Literal

Device = Literal["cpu", "cuda"]


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return True when an NVIDIA GPU is visible to the process."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def resolve_device(requested: str = "auto") -> Device:
    """Map ``auto`` / ``cpu`` / ``cuda`` to a concrete, available device."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available():
            raise RuntimeError("device='cuda' requested but no NVIDIA GPU is visible.")
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def gpu_name() -> str | None:
    """Human-readable name of the first GPU, if any."""
    if not cuda_available():
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
