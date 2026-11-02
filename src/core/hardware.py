"""Host hardware fingerprint recorded with every training run (reproducibility metadata)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    else:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            kernel32 = getattr(ctypes, "windll").kernel32  # windll exists only on Windows
            kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024**3, 1)
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return round(int(out.stdout.strip()) / 1024**3, 1)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except (OSError, ValueError):
        return None
    return None


@lru_cache(maxsize=1)
def machine_info() -> dict[str, Any]:
    """CPU, core count, RAM, OS and Python version of the current host."""
    return {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
    }


def machine_summary() -> str:
    """One-line human-readable hardware description."""
    m = machine_info()
    ram = f"{m['ram_gb']:.0f} GB RAM" if m["ram_gb"] else "RAM unknown"
    return f"{m['cpu']} · {m['logical_cores']} threads · {ram} · {m['os'].split(' (')[0]}"
