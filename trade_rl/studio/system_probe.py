"""Read-only local system telemetry for Studio."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from trade_rl.studio.contracts import SystemMetric, SystemSummary


def _cpu_metric() -> SystemMetric:
    count = max(os.cpu_count() or 1, 1)
    try:
        load = os.getloadavg()[0]
        value = min(max(load / count * 100.0, 0.0), 100.0)
        detail = f"load {load:.2f} / {count} cores"
    except (AttributeError, OSError):
        value = 0.0
        detail = f"{count} cores"
    return SystemMetric(label="CPU", value=value, detail=detail)


def _memory_metric() -> SystemMetric:
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
