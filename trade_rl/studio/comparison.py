"""Read-only comparison of canonical Studio run artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trade_rl.artifacts.run_manifest import validate_training_run_directory
from trade_rl.studio.contracts import (
    ComparisonMetric,
    ComparisonSeriesPoint,
    ConfigDifference,
    FoldComparison,
    RunComparison,
)

_METRICS: tuple[tuple[str, str, str], ...] = (
    ("total_return", "Total return", "higher"),
    ("sharpe", "Sharpe", "higher"),
    ("sortino", "Sortino", "higher"),
    ("max_drawdown", "Max drawdown", "lower"),
    ("turnover_total", "Turnover", "lower"),
    ("total_cost", "Total cost", "lower"),
    ("funding_pnl", "Funding PnL", "higher"),
    ("borrow_cost", "Borrow cost", "lower"),
)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Studio artifact must be a JSON object: {path.name}")
    return value


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _compound(values: object) -> float | None:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return None
    wealth = 1.0
    used = False
    for raw in values:
        value = _number(raw)
        if value is None:
            continue
        wealth *= 1.0 + value
        used = True
    return wealth - 1.0 if used else None


def _flatten(value: object, prefix: str = "") -> dict[str, str]:
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False, sort_keys=True)}
    if value is None:
        rendered = "null"
    elif isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, float):
        rendered = format(value, ".12g")
    else:
        rendered = str(value)
    return {prefix: rendered}


def _selected_metrics(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return {}
    return _mapping(payload.get("selected_metrics")) or {}


def _folds(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if payload is None or not isinstance(payload.get("folds"), list):
        return []
    return [item for item in payload["folds"] if isinstance(item, Mapping)]


def _returns(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    result: list[float] = []
    for fold in _folds(payload):
        values = fold.get(field)
        if not isinstance(values, list):
            continue
        result.extend(value for raw in values if (value := _number(raw)) is not None)
    return result


def _wealth(values: list[float]) -> list[float]:
    points = [1.0]
    current = 1.0
    for value in values:
        current *= 1.0 + value
        points.append(current)
    return points


def _at(values: list[float], index: int) -> float | None:
    return values[index] if index < len(values) else None


def compare_runs(left: Path, right: Path) -> RunComparison:
    """Validate and compare two immutable run directories."""

    left_manifest = validate_training_run_directory(left)
    right_manifest = validate_training_run_directory(right)
    left_walk = _read_json(left / "walk-forward.json")
    right_walk = _read_json(right / "walk-forward.json")
    left_metrics = _selected_metrics(left_walk)
    right_metrics = _selected_metrics(right_walk)

    metrics: list[ComparisonMetric] = []
    for key, label, preference in _METRICS:
        left_value = _number(left_metrics.get(key))
        right_value = _number(right_metrics.get(key))
        metrics.append(
            ComparisonMetric(
                key=key,
                label=label,
                left_value=left_value,
                right_value=right_value,
                delta=(
                    None
                    if left_value is None or right_value is None
                    else right_value - left_value
                ),
                preference=preference,
            )
        )

    left_config = _flatten(_read_json(left / "training-config.json") or {})
    right_config = _flatten(_read_json(right / "training-config.json") or {})
    config_differences = tuple(
        ConfigDifference(
            path=key, left=left_config.get(key), right=right_config.get(key)
        )
        for key in sorted(set(left_config) | set(right_config))
        if left_config.get(key) != right_config.get(key)
    )

    left_folds = {
        int(item.get("fold_index", index)): item
        for index, item in enumerate(_folds(left_walk))
    }
    right_folds = {
        int(item.get("fold_index", index)): item
        for index, item in enumerate(_folds(right_walk))
    }
    folds = tuple(
        FoldComparison(
            label=f"Fold {index + 1}",
            left_selected_return=_compound(
                left_folds.get(index, {}).get("selected_returns")
            ),
            left_baseline_return=_compound(
                left_folds.get(index, {}).get("baseline_returns")
            ),
            right_selected_return=_compound(
                right_folds.get(index, {}).get("selected_returns")
            ),
            right_baseline_return=_compound(
                right_folds.get(index, {}).get("baseline_returns")
            ),
        )
        for index in sorted(set(left_folds) | set(right_folds))
    )

    left_selected = _wealth(_returns(left_walk, "selected_returns"))
    right_selected = _wealth(_returns(right_walk, "selected_returns"))
    left_baseline = _wealth(_returns(left_walk, "baseline_returns"))
    right_baseline = _wealth(_returns(right_walk, "baseline_returns"))
    length = max(
        len(left_selected),
        len(right_selected),
        len(left_baseline),
        len(right_baseline),
    )
    wealth = tuple(
        ComparisonSeriesPoint(
            label=str(index),
            left=_at(left_selected, index),
            right=_at(right_selected, index),
            left_baseline=_at(left_baseline, index),
            right_baseline=_at(right_baseline, index),
        )
        for index in range(length)
    )

    return RunComparison(
        left_run_id=left_manifest.run_id,
        right_run_id=right_manifest.run_id,
        metrics=tuple(metrics),
        config_differences=config_differences,
        folds=folds,
        wealth=wealth,
    )


__all__ = ["compare_runs"]
