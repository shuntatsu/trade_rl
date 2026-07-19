"""Validated filesystem catalog used by the local Studio API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.data import load_market_dataset_artifact
from trade_rl.studio.contracts import (
    ActiveJob,
    ConfigSummary,
    DatasetSummary,
    EquityPoint,
    JobSummary,
    ProductionAssessment,
    RunSummary,
    StabilityFold,
    StudioAlert,
    StudioOverview,
    SystemMetric,
    SystemSummary,
)
from trade_rl.studio.settings import StudioSettings


def _iso_timestamp(value: np.datetime64) -> str:
    nanoseconds = int(value.astype("datetime64[ns]").astype(np.int64))
    return datetime.fromtimestamp(nanoseconds / 1_000_000_000, UTC).isoformat()


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _timeframe(bar_hours: float) -> str:
    if bar_hours < 1.0:
        return f"{int(round(bar_hours * 60.0))}m"
    if math.isclose(bar_hours % 24.0, 0.0):
        return f"{int(round(bar_hours / 24.0))}d"
    return f"{bar_hours:g}h"


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _invalid_id(path: Path) -> str:
    return hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()


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
        for line in path.read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0])
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
    return "CUDA unavailable", False, SystemMetric(label="GPU", value=0.0, detail="CUDA unavailable")


def _wealth_points(folds: object) -> tuple[EquityPoint, ...]:
    if not isinstance(folds, list):
        return ()
    selected: list[float] = []
    baseline: list[float] = []
    for raw_fold in folds:
        fold = _mapping(raw_fold)
        if fold is None:
            continue
        raw_selected = fold.get("selected_returns")
        raw_baseline = fold.get("baseline_returns")
        if isinstance(raw_selected, list) and isinstance(raw_baseline, list):
            selected.extend(
                value for item in raw_selected if (value := _number(item)) is not None
            )
            baseline.extend(
                value for item in raw_baseline if (value := _number(item)) is not None
            )
    size = min(len(selected), len(baseline))
    if size == 0:
        return ()
    rl_wealth = 1.0
    baseline_wealth = 1.0
    raw_points = [EquityPoint(label="0", rl=1.0, baseline=1.0)]
    for index in range(size):
        rl_wealth *= 1.0 + selected[index]
        baseline_wealth *= 1.0 + baseline[index]
        raw_points.append(
            EquityPoint(
                label=str(index + 1),
                rl=rl_wealth,
                baseline=baseline_wealth,
            )
        )
    if len(raw_points) <= 16:
        return tuple(raw_points)
    indices = np.linspace(0, len(raw_points) - 1, 16, dtype=int)
    return tuple(raw_points[int(index)] for index in indices)


def _stability_points(folds: object) -> tuple[StabilityFold, ...]:
    if not isinstance(folds, list):
        return ()
    points: list[StabilityFold] = []
    for index, raw_fold in enumerate(folds[:8]):
        fold = _mapping(raw_fold)
        if fold is None:
            continue
        selected_values = fold.get("selected_returns")
        baseline_values = fold.get("baseline_returns")
        if not isinstance(selected_values, list) or not isinstance(baseline_values, list):
            continue
        selected_wealth = 1.0
        baseline_wealth = 1.0
        for raw in selected_values:
            value = _number(raw)
            if value is not None:
                selected_wealth *= 1.0 + value
        for raw in baseline_values:
            value = _number(raw)
            if value is not None:
                baseline_wealth *= 1.0 + value
        selected_return = selected_wealth - 1.0
        baseline_return = baseline_wealth - 1.0
        points.append(
            StabilityFold(
                label=f"Fold {index + 1}",
                low=min(selected_return, baseline_return),
                median=selected_return,
                high=max(selected_return, baseline_return),
            )
        )
    return tuple(points)


class StudioCatalog:
    """Read and summarize canonical artifacts without mutating them."""

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _dataset_directories(self) -> tuple[Path, ...]:
        directories: set[Path] = set()
        for root in self.settings.dataset_roots:
            if (root / "manifest.json").is_file() and (root / "arrays.npz").is_file():
                directories.add(root)
            if not root.is_dir():
                continue
            for manifest in root.rglob("manifest.json"):
                if (manifest.parent / "arrays.npz").is_file():
                    directories.add(manifest.parent)
        return tuple(sorted(directories, key=lambda item: item.as_posix()))

    def list_datasets(self) -> tuple[DatasetSummary, ...]:
        records: list[DatasetSummary] = []
        for path in self._dataset_directories():
            relative = self.settings.relative_path(path)
            try:
                dataset = load_market_dataset_artifact(path)
                start = _iso_timestamp(dataset.timestamps[0])
                end = _iso_timestamp(dataset.timestamps[-1])
                records.append(
                    DatasetSummary(
                        id=dataset.dataset_id,
                        name=path.name,
                        relative_path=relative,
                        market=dataset.calendar_kind.value,
                        symbols=dataset.symbols,
                        timeframes=(_timeframe(dataset.bar_hours),),
                        range=f"{start} — {end}",
                        status="VALID",
                        feature_count=dataset.n_features,
                        bar_count=dataset.n_bars,
                        symbol_count=dataset.n_symbols,
                        updated=_mtime(path / "manifest.json"),
                    )
                )
            except (OSError, ValueError, TypeError) as error:
                records.append(
                    DatasetSummary(
                        id=_invalid_id(path),
                        name=path.name,
                        relative_path=relative,
                        market="unknown",
                        symbols=(),
                        timeframes=(),
                        range="—",
                        status="INVALID",
                        feature_count=0,
                        bar_count=0,
                        symbol_count=0,
                        updated=_mtime(path / "manifest.json"),
                        validation_error=str(error),
                    )
                )
        return tuple(sorted(records, key=lambda item: item.updated, reverse=True))

    def _run_directories(self) -> tuple[Path, ...]:
        directories: set[Path] = set()
        for root in self.settings.run_roots:
            runs_root = root / "runs"
            if not runs_root.is_dir():
                continue
            directories.update(path for path in runs_root.iterdir() if path.is_dir())
        return tuple(sorted(directories, key=lambda item: item.as_posix()))

    def _run_algorithm(self, path: Path) -> str:
        payload = _read_json(path / "training-config.json")
        training = None if payload is None else _mapping(payload.get("training"))
        algorithm = None if training is None else training.get("algorithm")
        return algorithm if isinstance(algorithm, str) and algorithm else "unknown"

    def _walk_forward_payload(self, path: Path) -> Mapping[str, Any] | None:
        return _read_json(path / "walk-forward.json")

    def list_runs(self) -> tuple[RunSummary, ...]:
        records: list[RunSummary] = []
        for path in self._run_directories():
            relative = self.settings.relative_path(path)
            try:
                manifest = validate_training_run_directory(path)
                walk_forward = self._walk_forward_payload(path)
                selected_metrics = (
                    None
                    if walk_forward is None
                    else _mapping(walk_forward.get("selected_metrics"))
                )
                records.append(
                    RunSummary(
                        id=manifest.run_id,
                        relative_path=relative,
                        run_kind=manifest.run_kind,
                        algorithm=self._run_algorithm(path),
                        dataset_id=manifest.dataset_id,
                        period=f"{manifest.created_at.date()} — {manifest.completed_at.date()}",
                        created_at=manifest.created_at.isoformat(),
                        completed_at=manifest.completed_at.isoformat(),
                        file_count=len(manifest.files),
                        sharpe=(
                            None
                            if selected_metrics is None
                            else _number(selected_metrics.get("sharpe"))
                        ),
                        max_drawdown=(
                            None
                            if selected_metrics is None
                            else _number(selected_metrics.get("max_drawdown"))
                        ),
                        total_return=(
                            None
                            if selected_metrics is None
                            else _number(selected_metrics.get("total_return"))
                        ),
                        production_status="NO-GO",
                        status="VALID",
                    )
                )
            except (OSError, ValueError, TypeError) as error:
                records.append(
                    RunSummary(
                        id=path.name,
                        relative_path=relative,
                        run_kind="unknown",
                        algorithm="unknown",
                        dataset_id="",
                        period="—",
                        created_at=_mtime(path),
                        completed_at=_mtime(path),
                        file_count=0,
                        production_status="NO-GO",
                        status="INVALID",
                        validation_error=str(error),
                    )
                )
        return tuple(sorted(records, key=lambda item: item.completed_at, reverse=True))

    def list_configs(self) -> tuple[ConfigSummary, ...]:
        records: list[ConfigSummary] = []
        paths: set[Path] = set()
        for root in self.settings.config_roots:
            if root.is_file() and root.suffix == ".json":
                paths.add(root)
            elif root.is_dir():
                paths.update(root.rglob("*.json"))
        for path in sorted(paths, key=lambda item: item.as_posix()):
            payload = _read_json(path)
            training = None if payload is None else _mapping(payload.get("training"))
            algorithm = None if training is None else training.get("algorithm")
            valid = payload is not None and training is not None
            records.append(
                ConfigSummary(
                    name=path.stem,
                    relative_path=self.settings.relative_path(path),
                    algorithm=(
                        algorithm
                        if isinstance(algorithm, str) and algorithm
                        else "unknown"
                    ),
                    status="VALID" if valid else "INVALID",
                    validation_error=None if valid else "training section is missing",
                )
            )
        return tuple(records)

    def _system(self) -> SystemSummary:
        gpu_name, cuda_ready, gpu_metric = _gpu_status()
        disk = shutil.disk_usage(self.settings.project_root)
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

    def overview(self, jobs: Sequence[JobSummary]) -> StudioOverview:
        datasets = self.list_datasets()
        runs = self.list_runs()
        valid_datasets = tuple(item for item in datasets if item.status == "VALID")
        valid_runs = tuple(item for item in runs if item.status == "VALID")
        active = tuple(
            ActiveJob(
                id=job.id,
                algorithm="training",
                phase=job.status,
                seed_progress=job.run_id,
                progress=100.0 if job.status == "succeeded" else 0.0,
            )
            for job in jobs
            if job.status in {"queued", "running", "cancelling"}
        )
        alerts: list[StudioAlert] = []
        if not valid_datasets:
            alerts.append(
                StudioAlert(level="warning", message="検証済みデータセットがありません", age="now")
            )
        invalid_dataset_count = len(datasets) - len(valid_datasets)
        if invalid_dataset_count:
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"無効なデータセットが{invalid_dataset_count}件あります",
                    age="now",
                )
            )
        if not valid_runs:
            alerts.append(
                StudioAlert(level="info", message="公開済みrunがありません", age="now")
            )
        invalid_run_count = len(runs) - len(valid_runs)
        if invalid_run_count:
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"無効なrunが{invalid_run_count}件あります",
                    age="now",
                )
            )
        if active:
            alerts.append(
                StudioAlert(
                    level="info",
                    message=f"{len(active)}件のジョブが実行中です",
                    age="now",
                )
            )
        while len(alerts) < 4:
            alerts.append(
                StudioAlert(
                    level="info",
                    message="ローカル研究モードで稼働しています",
                    age="now",
                )
            )
        latest_payload = None
        if valid_runs:
            run_path = self.settings.project_root / valid_runs[0].relative_path
            latest_payload = self._walk_forward_payload(run_path)
        equity = () if latest_payload is None else _wealth_points(latest_payload.get("folds"))
        stability = (
            () if latest_payload is None else _stability_points(latest_payload.get("folds"))
        )
        reasons = ["直接取引所への注文ルーティングは実装されていません"]
        if not valid_runs:
            reasons.append("検証済みrunがありません")
        elif valid_runs[0].sharpe is None:
            reasons.append("最新runにwalk-forward評価指標がありません")
        reasons.append("リリース承認とpaper reconciliationは未完了です")
        return StudioOverview(
            system=self._system(),
            latest_dataset=valid_datasets[0] if valid_datasets else None,
            active_jobs=active,
            runs=valid_runs[:4],
            alerts=tuple(alerts[:4]),
            equity=equity,
            stability=stability,
            assessment=ProductionAssessment(reasons=tuple(reasons)),
        )
