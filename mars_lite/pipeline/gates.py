from __future__ import annotations

import math
from typing import Any, Mapping

MetricMap = Mapping[str, float]


def diagnostic_baseline(name: str) -> bool:
    """Return True when a baseline is informative but never release-mandatory."""

    return name == "oracle_dp" or name.startswith("oracle_ic")


def evaluate_residual_alpha_gate(
    report: Mapping[str, object],
    *,
    threshold: float = 0.02,
    min_positive_ratio: float = 0.6,
    min_t_stat: float = 1.0,
) -> dict[str, Any]:
    """Apply the residual-alpha contract to a selected model's WF report."""

    model = str(report.get("model", ""))
    if model not in {"ridge", "gbm"}:
        raise ValueError("residual alpha report must identify ridge or gbm")
    try:
        mean_ic = float(report["mean_oos_ic"])
        positive_ratio = float(report["positive_fold_ratio"])
        t_stat = float(report["t_stat"])
        n_folds = int(report["n_folds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("residual alpha report is incomplete") from exc
    values = (
        mean_ic,
        positive_ratio,
        t_stat,
        threshold,
        min_positive_ratio,
        min_t_stat,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("residual alpha gate values must be finite")
    if n_folds <= 0:
        raise ValueError("residual alpha gate requires at least one OOS fold")
    checks = {
        "mean_oos_ic": mean_ic >= threshold,
        "positive_fold_ratio": positive_ratio >= min_positive_ratio,
        "stability": t_stat >= min_t_stat,
    }
    return {
        **dict(report),
        "model": model,
        "passed": all(checks.values()),
        "checks": checks,
        "threshold": threshold,
        "min_positive_ratio": min_positive_ratio,
        "min_t_stat": min_t_stat,
    }


def _finite_metric(metrics: MetricMap, key: str, label: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be finite")
    return value


def _finite_probability(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return value


def evaluate_direct_gate2(
    *,
    agent: MetricMap,
    baselines: Mapping[str, MetricMap],
) -> dict[str, Any]:
    """Evaluate the legacy direct policy against executable mandatory baselines only."""

    if "flat" not in baselines or "trend_following" not in baselines:
        raise ValueError("direct Gate 2 requires flat and trend_following baselines")
    agent_return = _finite_metric(agent, "total_return", "agent")
    details: dict[str, dict[str, object]] = {}
    for name, metrics in baselines.items():
        baseline_return = _finite_metric(metrics, "total_return", name)
        details[name] = {
            "rl_return": agent_return,
            "baseline_return": baseline_return,
            "rl_beat": agent_return > baseline_return,
            "mandatory": name in {"flat", "trend_following"},
            "diagnostic_only": name not in {"flat", "trend_following"},
        }
    checks = {
        "beats_flat": bool(details["flat"]["rl_beat"]),
        "beats_trend_following": bool(details["trend_following"]["rl_beat"]),
    }
    return {
        "passed": all(checks.values()),
        "candidate_mode": "direct_weights_v1",
        "mandatory_comparisons": ("flat", "trend_following"),
        "checks": checks,
        "rl_beat_trend_following": checks["beats_trend_following"],
        "details": details,
    }


def evaluate_residual_gate2(
    *,
    hybrid: MetricMap,
    shadow: MetricMap,
    flat: MetricMap,
    paired_p_value: float,
    diagnostic_results: Mapping[str, MetricMap] | None = None,
    max_drawdown_slack: float = 0.05,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a residual-RL candidate against executable mandatory references.

    Perfect/noisy oracles and alternative strategies can be attached for diagnosis,
    but their results never affect the pass decision.
    """

    hybrid_return = _finite_metric(hybrid, "total_return", "hybrid")
    hybrid_dd = _finite_metric(hybrid, "max_drawdown", "hybrid")
    shadow_return = _finite_metric(shadow, "total_return", "shadow")
    shadow_dd = _finite_metric(shadow, "max_drawdown", "shadow")
    flat_return = _finite_metric(flat, "total_return", "flat")
    p_value = _finite_probability(paired_p_value, "paired_p_value")
    if not math.isfinite(max_drawdown_slack) or max_drawdown_slack < 0.0:
        raise ValueError("max_drawdown_slack must be finite and non-negative")
    if not math.isfinite(significance_level) or not 0.0 < significance_level <= 1.0:
        raise ValueError("significance_level must be finite and within (0, 1]")

    checks = {
        "beats_flat": hybrid_return > flat_return,
        "beats_shadow": hybrid_return > shadow_return,
        "drawdown_within_slack": hybrid_dd <= shadow_dd + max_drawdown_slack,
        "paired_superiority_significant": p_value < significance_level,
    }
    diagnostics = {
        name: {
            "metrics": dict(metrics),
            "mandatory": False,
            "diagnostic_only": True,
            "oracle": diagnostic_baseline(name),
        }
        for name, metrics in (diagnostic_results or {}).items()
    }
    return {
        "passed": all(checks.values()),
        "candidate_mode": "ppo_residual_ensemble",
        "mandatory_comparisons": ("flat", "shadow"),
        "checks": checks,
        "paired_p_value": p_value,
        "diagnostic_results": diagnostics,
    }


def evaluate_baseline_only_gate(
    *,
    trend_development_gate: Mapping[str, object],
    holdout: MetricMap,
    cost2x_holdout: MetricMap,
    positive_return_p_value: float,
    max_drawdown_limit: float,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a pure executable trend baseline without self-comparison."""

    holdout_return = _finite_metric(holdout, "total_return", "holdout")
    holdout_dd = _finite_metric(holdout, "max_drawdown", "holdout")
    cost2x_return = _finite_metric(cost2x_holdout, "total_return", "cost2x_holdout")
    p_value = _finite_probability(positive_return_p_value, "positive_return_p_value")
    if not math.isfinite(max_drawdown_limit) or max_drawdown_limit < 0.0:
        raise ValueError("max_drawdown_limit must be finite and non-negative")
    if not math.isfinite(significance_level) or not 0.0 < significance_level <= 1.0:
        raise ValueError("significance_level must be finite and within (0, 1]")

    checks = {
        "trend_development_gate": trend_development_gate.get("passed") is True,
        "positive_holdout_return": holdout_return > 0.0,
        "drawdown_within_policy": holdout_dd <= max_drawdown_limit,
        "cost2x_non_negative": cost2x_return >= 0.0,
        "positive_return_significant": p_value < significance_level,
    }
    return {
        "passed": all(checks.values()),
        "candidate_mode": "baseline_only",
        "checks": checks,
        "positive_return_p_value": p_value,
    }
