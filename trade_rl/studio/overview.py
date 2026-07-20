"""Dashboard composition from focused Studio services."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
        alerts: list[StudioAlert] = []
        if not valid_datasets:
            alerts.append(
                StudioAlert(
                    level="warning",
                    message="検証済みデータセットがありません",
                    age="now",
                )
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
            alerts=tuple(alerts[:4]),
            equity=equity,
            stability=stability,
            assessment=ProductionAssessment(reasons=tuple(reasons)),
        )
