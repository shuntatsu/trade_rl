"""Read-only local system telemetry for Studio."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from trade_rl.studio.contracts import SystemMetric, SystemSummary

_CPU_LOCK = threading.Lock()
_CPU_SAMPLE: tuple[int, int] | None = None


def _windows_kernel32() -> Any | None:
    """Return kernel32 without assuming Windows-only ctypes attributes exist."""

    windll = getattr(ctypes, "windll", None)
    return None if windll is None else windll.kernel32


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _windows_cpu_percent() -> float | None:
    if platform.system() != "Windows":
        return None
    kernel32 = _windows_kernel32()
    if kernel32 is None:
        return None
    idle = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    if not kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
    current = (kernel.value + user.value, idle.value)
    global _CPU_SAMPLE
    with _CPU_LOCK:
        previous, _CPU_SAMPLE = _CPU_SAMPLE, current
    if previous is None:
        time.sleep(0.05)
        return _windows_cpu_percent()
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return min(max((total_delta - idle_delta) / total_delta * 100.0, 0.0), 100.0)


def _cpu_metric() -> SystemMetric:
    count = max(os.cpu_count() or 1, 1)
    windows_value = _windows_cpu_percent()
    if windows_value is not None:
        return SystemMetric(
            label="CPU", value=windows_value, detail=f"{count} logical cores"
        )
    try:
        load = os.getloadavg()[0]  # type: ignore[attr-defined]
        value = min(max(load / count * 100.0, 0.0), 100.0)
        detail = f"load {load:.2f} / {count} cores"
    except (AttributeError, OSError):
        value = 0.0
        detail = f"{count} cores"
    return SystemMetric(label="CPU", value=value, detail=detail)


def _memory_metric() -> SystemMetric:
    if platform.system() == "Windows":
        kernel32 = _windows_kernel32()
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if kernel32 is not None and kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            used = status.total_physical - status.available_physical
            return SystemMetric(
                label="メモリ",
                value=float(status.memory_load),
                detail=(
                    f"{used / 1024**3:.1f} / {status.total_physical / 1024**3:.1f} GB"
                ),
            )
    path = Path("/proc/meminfo")
    if path.is_file():
        values: dict[str, int] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                name, raw = line.split(":", 1)
                values[name] = int(raw.strip().split()[0])
        except (OSError, ValueError):
            values = {}
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        if total > 0:
            used = total - available
            return SystemMetric(
                label="メモリ",
                value=used / total * 100.0,
                detail=f"{used / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} GB",
            )
    return SystemMetric(label="メモリ", value=0.0, detail="利用情報なし")


def _gpu_status() -> tuple[str, bool, SystemMetric]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        if rows:
            parsed = [tuple(part.strip() for part in row.split(",")) for row in rows]
            if all(len(row) == 4 for row in parsed):
                names = [row[0] for row in parsed]
                utilization = max(float(row[1]) for row in parsed)
                used_mib = sum(float(row[2]) for row in parsed)
                total_mib = sum(float(row[3]) for row in parsed)
                name = names[0] if len(names) == 1 else f"{len(names)} GPUs"
                return (
                    name,
                    True,
                    SystemMetric(
                        label="GPU",
                        value=min(max(utilization, 0.0), 100.0),
                        detail=f"{used_mib / 1024:.1f} / {total_mib / 1024:.1f} GB",
                    ),
                )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0))
            allocated = float(torch.cuda.memory_allocated(0))
            total = float(torch.cuda.get_device_properties(0).total_memory)
            value = 0.0 if total <= 0.0 else allocated / total * 100.0
            return (
                name,
                True,
                SystemMetric(
                    label="GPU",
                    value=min(max(value, 0.0), 100.0),
                    detail=f"{allocated / 1024**3:.1f} / {total / 1024**3:.1f} GB",
                ),
            )
    except (ImportError, RuntimeError, AttributeError):
        pass
    return (
        "CUDA unavailable",
        False,
        SystemMetric(label="GPU", value=0.0, detail="CUDA unavailable"),
    )


class SystemProbe:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def snapshot(self) -> SystemSummary:
        gpu_name, cuda_ready, gpu_metric = _gpu_status()
        disk = shutil.disk_usage(self.project_root)
        disk_value = 0.0 if disk.total <= 0 else disk.used / disk.total * 100.0
        return SystemSummary(
            gpu_name=gpu_name,
            cuda_ready=cuda_ready,
            python_version=platform.python_version(),
            metrics=(
                gpu_metric,
                _cpu_metric(),
                _memory_metric(),
                SystemMetric(
                    label="ディスク",
                    value=disk_value,
                    detail=f"{disk.used / 1024**3:.0f} / {disk.total / 1024**3:.0f} GB",
                ),
            ),
        )
