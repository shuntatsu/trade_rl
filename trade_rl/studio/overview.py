"""Dashboard composition from focused Studio services."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from trade_rl.studio.catalog_common import read_json
from trade_rl.studio.contracts import (
    ActiveJob,
    EquityPoint,
    JobSummary,
    ProductionAssessment,
    StabilityFold,
    StudioAlert,
    StudioOverview,
)
from trade_rl.studio.dataset_catalog import DatasetCatalog
from trade_rl.studio.run_catalog import RunCatalog
from trade_rl.studio.system_probe import SystemProbe


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _relative_age(value: str | None, *, now: datetime) -> str:
    timestamp = _timestamp(value)
    if timestamp is None:
        return "時刻不明"
    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 5:
        return "たった今"
    if seconds < 60:
        return f"{seconds}秒前"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分前"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}時間前"
    return f"{hours // 24}日前"


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
            EquityPoint(label=str(index + 1), rl=rl_wealth, baseline=baseline_wealth)
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
        if not isinstance(selected_values, list) or not isinstance(
            baseline_values, list
        ):
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


def _supervised_active_jobs(runs: RunCatalog) -> tuple[ActiveJob, ...]:
    active: list[ActiveJob] = []
    now = datetime.now(UTC)
    for root in runs.settings.run_roots:
        if not root.is_dir():
            continue
        generation_root = root / "runs" if (root / "runs").is_dir() else root
        for generation in generation_root.iterdir():
            payload = read_json(generation / "heartbeat.json")
            if payload is None or payload.get("state") != "running":
                continue
            observed = payload.get("observed_at")
            try:
                age = (now - datetime.fromisoformat(str(observed))).total_seconds()
            except (TypeError, ValueError):
                continue
            if age < 0.0 or age > 120.0:
                continue
            phase = payload.get("phase")
            active.append(
                ActiveJob(
                    id=generation.name,
                    algorithm="full walk-forward",
                    phase=phase if isinstance(phase, str) else "running",
                    seed_progress="supervised Docker run",
                    progress=0.0,
                )
            )
    return tuple(active)


class OverviewService:
    def __init__(
        self,
        datasets: DatasetCatalog,
        runs: RunCatalog,
        system: SystemProbe,
    ) -> None:
        self.datasets = datasets
        self.runs = runs
        self.system = system

    def build(self, jobs: Sequence[JobSummary]) -> StudioOverview:
        now = datetime.now(UTC)
        datasets = self.datasets.list()
        runs = self.runs.list()
        valid_datasets = tuple(item for item in datasets if item.status == "VALID")
        valid_runs = tuple(item for item in runs if item.status == "VALID")
        active = tuple(
            ActiveJob(
                id=job.id,
                algorithm="training",
                phase=job.status,
                seed_progress=job.run_id,
                progress=0.0,
            )
            for job in jobs
            if job.status in {"queued", "running", "cancelling"}
        )
        known_ids = {item.id for item in active}
        active += tuple(
            item
            for item in _supervised_active_jobs(self.runs)
            if item.id not in known_ids
        )
        alerts: list[StudioAlert] = []
        if not valid_datasets:
            alerts.append(
                StudioAlert(
                    level="warning",
                    message="検証済みデータセットがありません",
                    age="現在",
                )
            )
        for dataset in datasets:
            if dataset.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"データセット {dataset.name} が無効です",
                    age=_relative_age(dataset.updated, now=now),
                )
            )
        if not valid_runs:
            alerts.append(
                StudioAlert(level="info", message="公開済みrunがありません", age="現在")
            )
        for run in runs:
            if run.status != "INVALID":
                continue
            alerts.append(
                StudioAlert(
                    level="warning",
                    message=f"run {run.run_id} が無効です",
                    age=_relative_age(run.completed_at or run.created_at, now=now),
                )
            )
        active_jobs = tuple(
            job for job in jobs if job.status in {"queued", "running", "cancelling"}
        )
        for job in active_jobs:
            alerts.append(
                StudioAlert(
                    level="info",
                    message=f"ジョブ {job.id} が実行中です",
                    age=_relative_age(job.started_at or job.submitted_at, now=now),
                )
            )
        fallback_active_count = len(active) - len(active_jobs)
        if fallback_active_count > 0:
            alerts.append(
                StudioAlert(
                    level="info",
                    message=f"{fallback_active_count}件の外部ジョブが実行中です",
                    age="現在",
                )
            )

        latest_payload = None
        if valid_runs:
            latest = self.runs.resolve(valid_runs[0].id)
            latest_payload = read_json(latest.path / "walk-forward.json")
        equity = (
            ()
            if latest_payload is None
            else _wealth_points(latest_payload.get("folds"))
        )
        stability = (
            ()
            if latest_payload is None
            else _stability_points(latest_payload.get("folds"))
        )
        reasons = ["直接取引所への注文ルーティングは実装されていません"]
        if not valid_runs:
            reasons.append("検証済みrunがありません")
        elif valid_runs[0].sharpe is None:
            reasons.append("最新runにwalk-forward評価指標がありません")
        reasons.append("リリース承認とpaper reconciliationは未完了です")
        return StudioOverview(
            system=self.system.snapshot(),
            latest_dataset=valid_datasets[0] if valid_datasets else None,
            active_jobs=active,
            runs=valid_runs[:4],
            alerts=tuple(alerts[:50]),
            equity=equity,
            stability=stability,
            assessment=ProductionAssessment(reasons=tuple(reasons)),
        )
