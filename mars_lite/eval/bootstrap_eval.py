"""Bootstrap statistical evaluation helpers for walk-forward reports."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mars_lite.utils.metrics import calc_sharpe_ratio


def bootstrap_sharpe_difference(
    candidate_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 1_000,
    ci: float = 0.95,
    seed: int | None = None,
    annualization_factor: float = 252,
) -> Dict[str, float | int]:
    """Estimate Sharpe difference confidence interval and two-sided p-value."""

    candidate = np.asarray(candidate_returns, dtype=float)
    baseline = np.asarray(baseline_returns, dtype=float)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate_returns and baseline_returns must match")
    if candidate.ndim != 1:
        raise ValueError("returns must be 1D arrays")
    if len(candidate) < 2:
        raise ValueError("at least two returns are required")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, len(candidate), size=len(candidate))
        diffs[idx] = _sharpe_diff(
            candidate[sample], baseline[sample], annualization_factor
        )

    alpha = 1.0 - ci
    quantiles = np.asarray(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))
    lower = float(quantiles[0])
    upper = float(quantiles[1])
    observed = _sharpe_diff(candidate, baseline, annualization_factor)
    p_value = float(np.mean(np.abs(diffs - np.mean(diffs)) >= abs(observed)))
    return {
        "mean": float(np.mean(diffs)),
        "lower_ci": lower,
        "upper_ci": upper,
        "p_value": p_value,
        "observed_diff": float(observed),
        "n_bootstrap": n_bootstrap,
        "ci": ci,
    }


def _sharpe_diff(
    candidate: np.ndarray, baseline: np.ndarray, annualization_factor: float
) -> float:
    return calc_sharpe_ratio(
        candidate, annualization_factor=annualization_factor
    ) - calc_sharpe_ratio(baseline, annualization_factor=annualization_factor)
