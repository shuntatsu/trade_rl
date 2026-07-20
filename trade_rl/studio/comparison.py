"""Read-only, eligibility-aware comparison of canonical Studio run artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from trade_rl.studio.contracts import (
    ComparisonEligibility,
    ComparisonMetric,
    ComparisonSeriesPoint,
    ConfigDifference,
    FoldComparison,
    RunComparison,
)
from trade_rl.studio.run_catalog import ResolvedRun

MetricPreference = Literal["higher", "lower", "neutral"]

_METRICS: tuple[tuple[str, str, MetricPreference], ...] = (
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
    folds = [item for item in payload["folds"] if isinstance(item, Mapping)]
    return sorted(folds, key=lambda item: int(item.get("fold_index", 0)))


def _return_length(fold: Mapping[str, Any], field: str) -> int | None:
    values = fold.get(field)
    return len(values) if isinstance(values, list) else None


def _test_range(fold: Mapping[str, Any]) -> tuple[int, int] | None:
    value = fold.get("test_range")
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value[0] < value[1]
    ):
        return int(value[0]), int(value[1])
    return None


def _eligibility(
    left: ResolvedRun,
    right: ResolvedRun,
    left_walk: Mapping[str, Any] | None,
    right_walk: Mapping[str, Any] | None,
) -> ComparisonEligibility:
    fatal: list[str] = []
    partial: list[str] = []
    if left.manifest.dataset_id != right.manifest.dataset_id:
        fatal.append("dataset identities differ")
    for side, resolved, payload in (
        ("left", left, left_walk),
        ("right", right, right_walk),
    ):
        if payload is None:
            fatal.append(f"{side} run has no walk-forward evidence")
            continue
        payload_dataset = payload.get("dataset_id")
        if (
            isinstance(payload_dataset, str)
            and payload_dataset != resolved.manifest.dataset_id
        ):
            fatal.append(
                f"{side} walk-forward dataset identity differs from its run manifest"
            )
    if left_walk is not None and right_walk is not None:
        if left_walk.get("schema_version") != right_walk.get("schema_version"):
            fatal.append("walk-forward schemas differ")
        left_stitch = left_walk.get("stitch_mode")
        right_stitch = right_walk.get("stitch_mode")
        if left_stitch is None or right_stitch is None:
            partial.append("stitch mode is unavailable")
        elif left_stitch != right_stitch:
            fatal.append("walk-forward stitch modes differ")
        left_folds = _folds(left_walk)
        right_folds = _folds(right_walk)
        left_indices = tuple(
            int(item.get("fold_index", index)) for index, item in enumerate(left_folds)
        )
        right_indices = tuple(
            int(item.get("fold_index", index)) for index, item in enumerate(right_folds)
        )
        if left_indices != right_indices:
            fatal.append("fold identities differ")
        else:
            left_lengths = tuple(
                (
                    _return_length(item, "selected_returns"),
                    _return_length(item, "baseline_returns"),
                )
                for item in left_folds
            )
            right_lengths = tuple(
                (
                    _return_length(item, "selected_returns"),
                    _return_length(item, "baseline_returns"),
                )
                for item in right_folds
            )
            if left_lengths != right_lengths:
                fatal.append("fold return lengths differ")
            left_ranges = tuple(_test_range(item) for item in left_folds)
            right_ranges = tuple(_test_range(item) for item in right_folds)
            if any(item is None for item in (*left_ranges, *right_ranges)):
                partial.append("sealed test ranges are unavailable")
            elif left_ranges != right_ranges:
                fatal.append("sealed test ranges differ")
    if fatal:
        return ComparisonEligibility(
            status="NOT_COMPARABLE",
            reasons=tuple(dict.fromkeys(fatal + partial)),
            dataset_id=(
                left.manifest.dataset_id
                if left.manifest.dataset_id == right.manifest.dataset_id
                else None
            ),
        )
    if partial:
        return ComparisonEligibility(
            status="PARTIALLY_COMPARABLE",
            reasons=tuple(dict.fromkeys(partial)),
            dataset_id=left.manifest.dataset_id,
        )
    return ComparisonEligibility(
        status="COMPARABLE",
        reasons=("dataset and sealed evaluation ranges align",),
        dataset_id=left.manifest.dataset_id,
    )


def _series(
    payload: Mapping[str, Any] | None, field: str
) -> tuple[list[str], list[float]]:
    labels = ["start"]
    values = [1.0]
    wealth = 1.0
    fallback_index = 0
    for fold in _folds(payload):
        returns = fold.get(field)
        if not isinstance(returns, list):
            continue
        test_range = _test_range(fold)
        for offset, raw in enumerate(returns):
            value = _number(raw)
            if value is None:
                continue
            wealth *= 1.0 + value
            label = (
                str(test_range[0] + offset)
                if test_range is not None
                else str(fallback_index)
            )
            fallback_index += 1
            labels.append(label)
            values.append(wealth)
    return labels, values


def _at(values: list[float], index: int) -> float | None:
    return values[index] if index < len(values) else None


def compare_runs(left: ResolvedRun, right: ResolvedRun) -> RunComparison:
    """Compare two validated runs only to the extent their evaluations align."""

    left_walk = _read_json(left.path / "walk-forward.json")
    right_walk = _read_json(right.path / "walk-forward.json")
    eligibility = _eligibility(left, right, left_walk, right_walk)

    left_config = _flatten(_read_json(left.path / "training-config.json") or {})
    right_config = _flatten(_read_json(right.path / "training-config.json") or {})
    config_differences = tuple(
        ConfigDifference(
            path=key, left=left_config.get(key), right=right_config.get(key)
        )
        for key in sorted(set(left_config) | set(right_config))
        if left_config.get(key) != right_config.get(key)
    )

    if eligibility.status == "NOT_COMPARABLE":
        return RunComparison(
            left_resource_id=left.summary.id,
            right_resource_id=right.summary.id,
            left_run_id=left.summary.run_id,
            right_run_id=right.summary.run_id,
            eligibility=eligibility,
            metrics=(),
            config_differences=config_differences,
            folds=(),
            wealth=(),
        )

    left_metrics = _selected_metrics(left_walk)
    right_metrics = _selected_metrics(right_walk)
    metrics = tuple(
        ComparisonMetric(
            key=key,
            label=label,
            left_value=(left_value := _number(left_metrics.get(key))),
            right_value=(right_value := _number(right_metrics.get(key))),
            delta=(
                None
                if left_value is None or right_value is None
                else right_value - left_value
            ),
            preference=preference,
        )
        for key, label, preference in _METRICS
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

    left_labels, left_selected = _series(left_walk, "selected_returns")
    right_labels, right_selected = _series(right_walk, "selected_returns")
    _, left_baseline = _series(left_walk, "baseline_returns")
    _, right_baseline = _series(right_walk, "baseline_returns")
    labels = left_labels if len(left_labels) >= len(right_labels) else right_labels
    length = max(
        len(left_selected), len(right_selected), len(left_baseline), len(right_baseline)
    )
    wealth = tuple(
        ComparisonSeriesPoint(
            label=labels[index] if index < len(labels) else str(index),
            left=_at(left_selected, index),
            right=_at(right_selected, index),
            left_baseline=_at(left_baseline, index),
            right_baseline=_at(right_baseline, index),
        )
        for index in range(length)
    )

    return RunComparison(
        left_resource_id=left.summary.id,
        right_resource_id=right.summary.id,
        left_run_id=left.summary.run_id,
        right_run_id=right.summary.run_id,
        eligibility=eligibility,
        metrics=metrics,
        config_differences=config_differences,
        folds=folds,
        wealth=wealth,
    )


__all__ = ["compare_runs"]
