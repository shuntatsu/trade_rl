"""Execution and result-blind decision helpers for Experiment 0002."""

from __future__ import annotations

import math

import numpy as np

from trade_rl.evaluation.experiments import ExperimentDecisionKind

EXPECTED_BASELINE_MEDIAN_TURNOVER = 858.3114067468092


def _compound(values: np.ndarray) -> float:
    """Reconstruct ordered compounded total return from raw interval returns."""

    wealth = 1.0
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= -1.0:
            raise RuntimeError("raw return is non-finite or <= -1")
        wealth *= 1.0 + resolved
    result = wealth - 1.0
    if not math.isfinite(result):
        raise RuntimeError("compounded return is non-finite")
    return result


def _formal_decision(
    *,
    positive_effects: int,
    median_excess: float,
    candidate_positive_returns: int,
    candidate_median_turnover: float,
) -> ExperimentDecisionKind:
    """Apply the immutable Experiment 0002 preregistered decision rule."""

    accept = (
        positive_effects >= 4
        and median_excess > 0.0
        and candidate_positive_returns == 5
        and candidate_median_turnover < EXPECTED_BASELINE_MEDIAN_TURNOVER
    )
    keep_baseline = (
        positive_effects <= 2
        or median_excess <= 0.0
        or candidate_positive_returns <= 3
    )
    if accept:
        return ExperimentDecisionKind.ACCEPT_CANDIDATE
    if keep_baseline:
        return ExperimentDecisionKind.KEEP_BASELINE
    return ExperimentDecisionKind.INCONCLUSIVE


__all__ = ["_compound", "_formal_decision"]
