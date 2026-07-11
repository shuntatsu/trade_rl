"""Bootstrap statistical evaluation helpers for walk-forward reports."""

from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np

from mars_lite.utils.metrics import calc_sharpe_ratio


def bootstrap_sharpe_difference(
    candidate_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 1_000,
    ci: float = 0.95,
    seed: int | None = None,
    annualization_factor: float = 252,
    block_size: int | None = None,
    method: Literal["moving_block", "stationary", "iid"] = "moving_block",
) -> Dict[str, Any]:
    """Estimate Sharpe difference confidence interval and null-hypothesis p-value

    Using Moving Block Bootstrap or Stationary Bootstrap to preserve autocorrelation
    and volatility clustering, with centered bootstrap p-value computation.
    """
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

    n = len(candidate)
    b = block_size if block_size is not None and block_size > 0 else max(1, min(n, int(np.ceil(n ** 0.5))))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample_indices = _generate_bootstrap_indices(n, b, method, rng)
        diffs[idx] = _sharpe_diff(
            candidate[sample_indices],
            baseline[sample_indices],
            annualization_factor,
        )

    alpha = 1.0 - ci
    quantiles = np.asarray(np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))
    lower = float(quantiles[0])
    upper = float(quantiles[1])
    observed = _sharpe_diff(candidate, baseline, annualization_factor)

    # 帰無仮説 (差分 = 0) 下に中心化した分布における両側 p-value
    if abs(observed) < 1e-12:
        p_value = 1.0
    else:
        centered_diffs = diffs - np.mean(diffs)
        p_value = float(np.mean(np.abs(centered_diffs) >= abs(observed)))

    return {
        "mean": float(np.mean(diffs)),
        "lower_ci": lower,
        "upper_ci": upper,
        "p_value": p_value,
        "observed_diff": float(observed),
        "n_bootstrap": n_bootstrap,
        "ci": ci,
        "block_size": int(b),
        "method": method,
    }


def analyze_block_size_sensitivity(
    candidate_returns: np.ndarray,
    baseline_returns: np.ndarray,
    block_sizes: list[int] | None = None,
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Sensitivity analysis across multiple block sizes."""
    if block_sizes is None:
        block_sizes = [1, 5, 10, 20]
    results = {}
    for b in block_sizes:
        results[b] = bootstrap_sharpe_difference(
            candidate_returns,
            baseline_returns,
            n_bootstrap=n_bootstrap,
            block_size=b,
            seed=seed,
        )
    return results


def _generate_bootstrap_indices(
    n: int, b: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    if method == "iid" or b <= 1:
        return rng.integers(0, n, size=n)

    indices = np.empty(n, dtype=int)
    pos = 0
    if method == "stationary":
        p = 1.0 / max(b, 1)
        while pos < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            for k in range(length):
                if pos >= n:
                    break
                indices[pos] = (start + k) % n
                pos += 1
        return indices

    # moving_block (default)
    while pos < n:
        max_start = max(1, n - b + 1)
        start = rng.integers(0, max_start)
        length = min(b, n - pos)
        indices[pos : pos + length] = np.arange(start, start + length)
        pos += length
    return indices


def _sharpe_diff(
    candidate: np.ndarray, baseline: np.ndarray, annualization_factor: float
) -> float:
    return calc_sharpe_ratio(
        candidate, annualization_factor=annualization_factor
    ) - calc_sharpe_ratio(baseline, annualization_factor=annualization_factor)

