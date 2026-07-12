from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualFoldSpec:
    fold: int
    outer_train_start: int
    outer_train_end: int
    inner_train_start: int
    inner_train_end: int
    inner_validation_start: int
    inner_validation_end: int
    outer_test_start: int
    outer_test_end: int
    purge_bars: int

    @property
    def inner_train_bars(self) -> int:
        return self.inner_train_end - self.inner_train_start

    @property
    def inner_validation_bars(self) -> int:
        return self.inner_validation_end - self.inner_validation_start

    @property
    def outer_test_bars(self) -> int:
        return self.outer_test_end - self.outer_test_start


def build_residual_fold_specs(
    *,
    n_bars: int,
    n_folds: int,
    purge_bars: int,
    horizon: int,
) -> tuple[list[ResidualFoldSpec], list[dict[str, object]]]:
    if min(n_bars, n_folds, purge_bars, horizon) <= 0:
        raise ValueError("n_bars, n_folds, purge_bars, and horizon must be positive")

    effective_purge = max(int(purge_bars), int(horizon), 24)
    edges = np.linspace(int(n_bars * 0.4), n_bars, n_folds + 1).astype(int)
    specs: list[ResidualFoldSpec] = []
    skipped: list[dict[str, object]] = []

    for fold in range(n_folds):
        outer_train_end = int(edges[fold])
        outer_test_start = outer_train_end + effective_purge
        outer_test_end = int(edges[fold + 1])
        inner_train_end = int(outer_train_end * 0.8)
        inner_validation_start = inner_train_end + effective_purge
        inner_validation_end = outer_train_end

        sizes = {
            "inner_train_too_short": inner_train_end,
            "inner_validation_too_short": inner_validation_end
            - inner_validation_start,
            "outer_test_too_short": outer_test_end - outer_test_start,
        }
        reason = next(
            (
                name
                for name, value in sizes.items()
                if value
                < {
                    "inner_train_too_short": 200,
                    "inner_validation_too_short": 100,
                    "outer_test_too_short": 50,
                }[name]
            ),
            None,
        )
        if reason is not None:
            skipped.append(
                {
                    "fold": fold,
                    "reason": reason,
                    "outer_train_end": outer_train_end,
                    "inner_train_end": inner_train_end,
                    "inner_validation_start": inner_validation_start,
                    "inner_validation_end": inner_validation_end,
                    "outer_test_start": outer_test_start,
                    "outer_test_end": outer_test_end,
                    "purge_bars": effective_purge,
                }
            )
            continue

        specs.append(
            ResidualFoldSpec(
                fold=fold,
                outer_train_start=0,
                outer_train_end=outer_train_end,
                inner_train_start=0,
                inner_train_end=inner_train_end,
                inner_validation_start=inner_validation_start,
                inner_validation_end=inner_validation_end,
                outer_test_start=outer_test_start,
                outer_test_end=outer_test_end,
                purge_bars=effective_purge,
            )
        )

    return specs, skipped


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_residual_folds(
    folds: list[dict[str, object]],
    *,
    requested_folds: int,
    skipped_folds: list[dict[str, object]],
) -> dict[str, object]:
    selection_counts = {"A": 0, "B": 0, "D": 0}
    hybrid_returns_1x: list[float] = []
    hybrid_returns_2x: list[float] = []
    shadow_returns_1x: list[float] = []
    shadow_returns_2x: list[float] = []
    excess_1x: list[float] = []
    excess_2x: list[float] = []
    fallback_count = 0
    member_count = 0
    alpha_enabled_count = 0
    hybrid_zero_trade_folds = 0
    shadow_zero_trade_folds = 0
    total_scored_oos_bars = 0

    for fold in folds:
        selected = str(fold["selected_configuration"])
        if selected in selection_counts:
            selection_counts[selected] += 1
        if bool(fold.get("alpha_enabled", False)):
            alpha_enabled_count += 1

        fallback_flags = [bool(value) for value in fold.get("selected_seed_fallbacks", [])]
        fallback_count += sum(fallback_flags)
        member_count += len(fallback_flags)

        outer_oos = fold["outer_oos"]
        if not isinstance(outer_oos, dict):
            raise TypeError("outer_oos must be a mapping")
        relative_1x = outer_oos["relative_1x"]
        relative_2x = outer_oos["relative_2x"]
        if not isinstance(relative_1x, dict) or not isinstance(relative_2x, dict):
            raise TypeError("relative evaluations must be mappings")

        hybrid_1x = relative_1x["hybrid"]
        shadow_1x = relative_1x["shadow"]
        paired_1x = relative_1x["paired"]
        hybrid_2x = relative_2x["hybrid"]
        shadow_2x = relative_2x["shadow"]
        paired_2x = relative_2x["paired"]
        for value in (
            hybrid_1x,
            shadow_1x,
            paired_1x,
            hybrid_2x,
            shadow_2x,
            paired_2x,
        ):
            if not isinstance(value, dict):
                raise TypeError("relative metric sections must be mappings")

        hybrid_returns_1x.append(float(hybrid_1x["total_return"]))
        hybrid_returns_2x.append(float(hybrid_2x["total_return"]))
        shadow_returns_1x.append(float(shadow_1x["total_return"]))
        shadow_returns_2x.append(float(shadow_2x["total_return"]))
        excess_1x.append(float(paired_1x["excess_log_return"]))
        excess_2x.append(float(paired_2x["excess_log_return"]))

        hybrid_zero_trade_folds += int(int(hybrid_1x["n_trades"]) == 0)
        shadow_zero_trade_folds += int(int(shadow_1x["n_trades"]) == 0)

        split = fold["split"]
        if not isinstance(split, dict):
            raise TypeError("split must be a mapping")
        total_scored_oos_bars += int(split["outer_test_scored_bars"])

    completed = len(folds)
    return {
        "requested_folds": int(requested_folds),
        "completed_folds": completed,
        "skipped_folds": len(skipped_folds),
        "selection_counts": selection_counts,
        "alpha_enabled_folds": alpha_enabled_count,
        "selected_member_fallback_count": fallback_count,
        "selected_member_count": member_count,
        "selected_member_fallback_rate": (
            float(fallback_count / member_count) if member_count else 0.0
        ),
        "hybrid_zero_trade_folds": hybrid_zero_trade_folds,
        "shadow_zero_trade_folds": shadow_zero_trade_folds,
        "hybrid_return_1x": _stats(hybrid_returns_1x),
        "hybrid_return_2x": _stats(hybrid_returns_2x),
        "shadow_return_1x": _stats(shadow_returns_1x),
        "shadow_return_2x": _stats(shadow_returns_2x),
        "paired_excess_log_return_1x": _stats(excess_1x),
        "paired_excess_log_return_2x": _stats(excess_2x),
        "hybrid_beats_shadow_fraction_1x": (
            float(sum(value > 0.0 for value in excess_1x) / completed)
            if completed
            else 0.0
        ),
        "hybrid_beats_shadow_fraction_2x": (
            float(sum(value > 0.0 for value in excess_2x) / completed)
            if completed
            else 0.0
        ),
        "survives_cost2x_fraction": (
            float(sum(value >= 0.0 for value in excess_2x) / completed)
            if completed
            else 0.0
        ),
        "total_scored_oos_bars": total_scored_oos_bars,
    }


def save_residual_walk_forward_report(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    output.write_text(text + "\n", encoding="utf-8")
