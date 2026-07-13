"""Deterministic moving-block bootstrap for paired excess returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """One-sided significance and interval summary for a mean difference."""

    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int


def moving_block_mean_test(
    differences: tuple[float, ...],
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> BootstrapResult:
    """Estimate uncertainty while preserving short-range serial dependence."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if len(differences) < 2 or all(abs(value) <= 1e-15 for value in differences):
        return BootstrapResult(
            p_value=1.0,
            lower_ci=0.0,
            upper_ci=0.0,
            block_size=1,
        )

    values = np.asarray(differences, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("bootstrap differences must be finite")

    observed = fmean(differences)
    n_values = len(differences)
    block_size = max(1, min(n_values, math.ceil(math.sqrt(n_values))))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)

    for draw in range(n_bootstrap):
        sampled: list[int] = []
        while len(sampled) < n_values:
            maximum_start = max(1, n_values - block_size + 1)
            start = int(rng.integers(0, maximum_start))
            sampled.extend(range(start, min(start + block_size, n_values)))
        means[draw] = float(values[np.asarray(sampled[:n_values])].mean())

    lower, upper = np.quantile(means, [0.025, 0.975])
    if observed <= 0.0:
        p_value = 1.0
    else:
        centered = means - means.mean()
        p_value = float(np.mean(centered >= observed))

    return BootstrapResult(
        p_value=p_value,
        lower_ci=float(lower),
        upper_ci=float(upper),
        block_size=block_size,
    )
