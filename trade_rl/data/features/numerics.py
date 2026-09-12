"""CPU-portable scalar numerics for identity-bound market features."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from trade_rl.data.contracts import PORTABLE_FEATURE_NUMERICS_SCHEMA


def _sample(values: np.ndarray | Sequence[float], *, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    return array


def _non_empty(values: np.ndarray | Sequence[float], *, field: str) -> np.ndarray:
    array = _sample(values, field=field)
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    return array


def portable_log_scalar(value: float) -> float:
    """Return one natural logarithm using the scalar platform libm path."""

    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("portable log input must be finite")
    if resolved <= 0.0:
        raise ValueError("portable log input must be positive")
    return math.log(resolved)


def portable_log(values: np.ndarray) -> np.ndarray:
    """Return natural logarithms in deterministic C-order scalar evaluation."""

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("portable log input must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError("portable log input must contain only positive values")
    flattened = array.reshape(-1, order="C")
    logged = np.fromiter(
        (math.log(float(value)) for value in flattened),
        dtype=np.float64,
        count=flattened.size,
    )
    return logged.reshape(array.shape, order="C")


def portable_sum(values: np.ndarray | Sequence[float]) -> float:
    """Sum one finite one-dimensional sequence in fixed order using ``math.fsum``."""

    array = _sample(values, field="portable sum input")
    return math.fsum(float(value) for value in array)


def portable_mean(values: np.ndarray | Sequence[float]) -> float:
    """Return the population mean of one finite non-empty sequence."""

    array = _non_empty(values, field="portable mean input")
    return math.fsum(float(value) for value in array) / float(array.size)


def portable_variance(values: np.ndarray | Sequence[float]) -> float:
    """Return population variance using one fixed mean and summation order."""

    array = _non_empty(values, field="portable variance input")
    center = portable_mean(array)
    return math.fsum((float(value) - center) ** 2 for value in array) / float(
        array.size
    )


def portable_std(values: np.ndarray | Sequence[float]) -> float:
    """Return population standard deviation under portable variance semantics."""

    return math.sqrt(portable_variance(values))


def _paired_samples(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
    *,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_array = _non_empty(left, field=f"{field} left input")
    right_array = _non_empty(right, field=f"{field} right input")
    if left_array.shape != right_array.shape:
        raise ValueError(f"{field} inputs must have identical shape")
    return left_array, right_array


def portable_dot(
    left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]
) -> float:
    """Return a fixed-order dot product for equal non-empty finite sequences."""

    left_array, right_array = _paired_samples(left, right, field="portable dot")
    return math.fsum(
        float(left_value) * float(right_value)
        for left_value, right_value in zip(left_array, right_array, strict=True)
    )


def portable_covariance(
    left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]
) -> float:
    """Return population covariance using fixed-order portable reductions."""

    left_array, right_array = _paired_samples(
        left, right, field="portable covariance"
    )
    left_mean = portable_mean(left_array)
    right_mean = portable_mean(right_array)
    return math.fsum(
        (float(left_value) - left_mean) * (float(right_value) - right_mean)
        for left_value, right_value in zip(left_array, right_array, strict=True)
    ) / float(left_array.size)


def portable_correlation(
    left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]
) -> float:
    """Return population correlation for two non-degenerate finite sequences."""

    left_array, right_array = _paired_samples(
        left, right, field="portable correlation"
    )
    left_variance = portable_variance(left_array)
    right_variance = portable_variance(right_array)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0.0:
        raise ValueError("portable correlation requires non-zero variance")
    return portable_covariance(left_array, right_array) / denominator


__all__ = [
    "PORTABLE_FEATURE_NUMERICS_SCHEMA",
    "portable_correlation",
    "portable_covariance",
    "portable_dot",
    "portable_log",
    "portable_log_scalar",
    "portable_mean",
    "portable_std",
    "portable_sum",
    "portable_variance",
]
