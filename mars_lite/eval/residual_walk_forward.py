from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from mars_lite.eval.relative_evaluation import _moving_block_mean_test


@dataclass(frozen=True)
class ResidualFoldSpec:
    fold: int
    policy_train_start: int
    policy_train_end: int
    checkpoint_validation_start: int
    checkpoint_validation_end: int
    configuration_selection_start: int
    configuration_selection_end: int
    outer_test_start: int
    outer_test_end: int
    purge_bars: int

    @property
    def outer_train_start(self) -> int:
        return self.policy_train_start

    @property
    def outer_train_end(self) -> int:
        return self.configuration_selection_end

    @property
    def inner_train_start(self) -> int:
        return self.policy_train_start

    @property
    def inner_train_end(self) -> int:
        return self.policy_train_end

    @property
    def inner_validation_start(self) -> int:
        return self.configuration_selection_start

    @property
    def inner_validation_end(self) -> int:
        return self.configuration_selection_end

    @property
    def inner_train_bars(self) -> int:
        return self.policy_train_end - self.policy_train_start

    @property
    def checkpoint_validation_bars(self) -> int:
        return self.checkpoint_validation_end - self.checkpoint_validation_start

    @property
    def configuration_selection_bars(self) -> int:
        return self.configuration_selection_end - self.configuration_selection_start

    @property
    def inner_validation_bars(self) -> int:
        return self.configuration_selection_bars

    @property
    def outer_test_bars(self) -> int:
        return self.outer_test_end - self.outer_test_start


@dataclass(frozen=True)
class RelativeFoldSeries:
    fold: int
    hybrid_returns: np.ndarray
    shadow_returns: np.ndarray
    hybrid_trades: int
    shadow_trades: int
    hybrid_turnover: float
    shadow_turnover: float
    hybrid_cost: float
    shadow_cost: float


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
        available_development = outer_train_end - 2 * effective_purge

        if available_development > 0:
            policy_bars = int(available_development * 0.70)
            checkpoint_bars = int(available_development * 0.15)
            configuration_bars = available_development - policy_bars - checkpoint_bars
        else:
            policy_bars = checkpoint_bars = configuration_bars = 0

        policy_train_start = 0
        policy_train_end = policy_bars
        checkpoint_validation_start = policy_train_end + effective_purge
        checkpoint_validation_end = checkpoint_validation_start + checkpoint_bars
        configuration_selection_start = checkpoint_validation_end + effective_purge
        configuration_selection_end = configuration_selection_start + configuration_bars

        sizes = {
            "inner_train_too_short": policy_bars,
            "checkpoint_validation_too_short": checkpoint_bars,
            "inner_validation_too_short": configuration_bars,
            "outer_test_too_short": outer_test_end - outer_test_start,
        }
        minimums = {
            "inner_train_too_short": 200,
            "checkpoint_validation_too_short": 100,
            "inner_validation_too_short": 100,
            "outer_test_too_short": 50,
        }
        reason = next(
            (name for name, value in sizes.items() if value < minimums[name]),
            None,
        )
        if reason is not None:
            skipped.append(
                {
                    "fold": fold,
                    "reason": reason,
                    "outer_train_end": outer_train_end,
                    "policy_train_end": policy_train_end,
                    "checkpoint_validation_start": checkpoint_validation_start,
                    "checkpoint_validation_end": checkpoint_validation_end,
                    "configuration_selection_start": configuration_selection_start,
                    "configuration_selection_end": configuration_selection_end,
                    "outer_test_start": outer_test_start,
                    "outer_test_end": outer_test_end,
                    "purge_bars": effective_purge,
                }
            )
            continue

        specs.append(
            ResidualFoldSpec(
                fold=fold,
                policy_train_start=policy_train_start,
                policy_train_end=policy_train_end,
                checkpoint_validation_start=checkpoint_validation_start,
                checkpoint_validation_end=checkpoint_validation_end,
                configuration_selection_start=configuration_selection_start,
                configuration_selection_end=configuration_selection_end,
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


def _return_metrics(
    returns: np.ndarray,
    *,
    bars_per_year: int,
    n_trades: int,
    turnover: float,
    cost: float,
) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("return series must be one-dimensional")
    if bars_per_year <= 0:
        raise ValueError("bars_per_year must be positive")
    if np.any(~np.isfinite(values)) or np.any(values <= -1.0):
        raise ValueError("return series must be finite and greater than -1")

    std = float(values.std()) if values.size else 0.0
    sharpe = (
        float(values.mean() / std * np.sqrt(bars_per_year)) if std > 0.0 else 0.0
    )
    downside = np.minimum(values, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2))) if values.size else 0.0
    sortino = (
        float(values.mean() / downside_std * np.sqrt(bars_per_year))
        if downside_std > 0.0
        else 0.0
    )
    equity = np.concatenate(
        [np.ones(1, dtype=np.float64), np.cumprod(1.0 + values)]
    )
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(1.0 - equity / peak)) if equity.size else 0.0
    return {
        "total_return": float(equity[-1] - 1.0),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "n_trades": int(n_trades),
        "turnover_total": float(turnover),
        "total_cost": float(cost),
        "n_base_bars": int(values.size),
    }


def stitch_relative_fold_results(
    fold_series: list[RelativeFoldSeries],
    *,
    bars_per_year: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not fold_series:
        raise ValueError("at least one fold series is required")
    ordered = sorted(fold_series, key=lambda item: item.fold)
    if len({item.fold for item in ordered}) != len(ordered):
        raise ValueError("fold identifiers must be unique")

    hybrid_parts: list[np.ndarray] = []
    shadow_parts: list[np.ndarray] = []
    for item in ordered:
        hybrid = np.asarray(item.hybrid_returns, dtype=np.float64)
        shadow = np.asarray(item.shadow_returns, dtype=np.float64)
        if hybrid.ndim != 1 or shadow.ndim != 1 or hybrid.shape != shadow.shape:
            raise ValueError("hybrid and shadow fold returns must be aligned vectors")
        hybrid_parts.append(hybrid)
        shadow_parts.append(shadow)

    hybrid_returns = np.concatenate(hybrid_parts)
    shadow_returns = np.concatenate(shadow_parts)
    differences = hybrid_returns - shadow_returns
    hybrid = _return_metrics(
        hybrid_returns,
        bars_per_year=bars_per_year,
        n_trades=sum(item.hybrid_trades for item in ordered),
        turnover=sum(item.hybrid_turnover for item in ordered),
        cost=sum(item.hybrid_cost for item in ordered),
    )
    shadow = _return_metrics(
        shadow_returns,
        bars_per_year=bars_per_year,
        n_trades=sum(item.shadow_trades for item in ordered),
        turnover=sum(item.shadow_turnover for item in ordered),
        cost=sum(item.shadow_cost for item in ordered),
    )
    return {
        "fold_count": len(ordered),
        "n_base_bars": int(hybrid_returns.size),
        "hybrid": hybrid,
        "shadow": shadow,
        "paired": {
            "excess_total_return": float(hybrid["total_return"])
            - float(shadow["total_return"]),
            "excess_log_return": float(
                np.log1p(hybrid_returns).sum() - np.log1p(shadow_returns).sum()
            ),
            "mean_base_bar_excess": float(differences.mean())
            if differences.size
            else 0.0,
            **_moving_block_mean_test(differences, seed=bootstrap_seed),
        },
        "annualization_factor": int(bars_per_year),
    }


def summarize_residual_folds(
    folds: list[dict[str, object]],
    *,
    requested_folds: int,
    skipped_folds: list[dict[str, object]],
    stitched_oos: dict[str, object] | None = None,
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

        raw_fallbacks = fold.get("selected_seed_fallbacks", [])
        if not isinstance(raw_fallbacks, list):
            raise TypeError("selected_seed_fallbacks must be a list")
        fallback_flags = [bool(value) for value in raw_fallbacks]
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
    summary: dict[str, object] = {
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
    if stitched_oos is not None:
        summary["stitched_oos"] = stitched_oos
    return summary


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
