"""Deterministic moving-block bootstrap for paired excess returns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from statistics import fmean

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """One-sided significance and interval summary for a mean difference."""

    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int


def _resolve_block_size(*, requested: object, n_values: int) -> int:
    if requested is None:
        return max(1, min(n_values, math.ceil(math.sqrt(n_values))))
    value = _require_integer(requested, name="block_size", minimum=1)
    return min(value, max(n_values, 1))


def _require_integer(value: object, *, name: str, minimum: int) -> int:
    requirement = "positive" if minimum else "non-negative"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a {requirement} integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be a {requirement} integer")
    return resolved


def moving_block_mean_test(
    differences: tuple[float, ...],
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
    block_size: int | None = None,
) -> BootstrapResult:
    """Estimate uncertainty while preserving short-range serial dependence.

    ``block_size`` is optional for existing callers. Research protocols that
    predeclare a temporal block length can pass it explicitly; the effective
    size is capped only when the supplied sample is shorter than that block.
    """

    n_bootstrap = _require_integer(n_bootstrap, name="n_bootstrap", minimum=1)
    seed = _require_integer(seed, name="seed", minimum=0)
    effective_block_size = _resolve_block_size(
        requested=block_size,
        n_values=len(differences),
    )
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap differences must be a finite 1-D series")

    if len(differences) < 2 or all(abs(value) <= 1e-15 for value in differences):
        return BootstrapResult(
            p_value=1.0,
            lower_ci=0.0,
            upper_ci=0.0,
            block_size=(1 if block_size is None else effective_block_size),
        )

    try:
        observed = fmean(differences)
    except OverflowError as exc:
        raise ValueError("observed bootstrap mean must be finite") from exc
    if not math.isfinite(observed):
        raise ValueError("observed bootstrap mean must be finite")
    n_values = len(differences)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)

    # Preserve the historical resample stream and arithmetic for valid inputs,
    # but never publish an interval or p-value derived from overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        for draw in range(n_bootstrap):
            sampled: list[int] = []
            while len(sampled) < n_values:
                start = int(rng.integers(0, n_values))
                sampled.extend(
                    (start + offset) % n_values
                    for offset in range(effective_block_size)
                )
            means[draw] = float(values[np.asarray(sampled[:n_values])].mean())

        if not np.isfinite(means).all():
            raise ValueError("bootstrap resampled means must be finite")
        lower, upper = np.quantile(means, [0.025, 0.975])
        if not np.isfinite([lower, upper]).all():
            raise ValueError("bootstrap interval must be finite")
        if observed <= 0.0:
            p_value = 1.0
        else:
            centered = means - means.mean()
            if not np.isfinite(centered).all():
                raise ValueError("centered bootstrap means must be finite")
            extreme = int(np.count_nonzero(centered >= observed))
            p_value = float((extreme + 1) / (n_bootstrap + 1))

    return BootstrapResult(
        p_value=p_value,
        lower_ci=float(lower),
        upper_ci=float(upper),
        block_size=effective_block_size,
    )
