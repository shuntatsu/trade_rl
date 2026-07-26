"""Pure advantage diagnostics and raw Lagrangian composition utilities."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DEFAULT_EPSILON: Final[float] = 1e-8


def _validated_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite number")
    value = float(epsilon)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("epsilon must be a positive finite number")
    return value


def _finite_float_array(
    values: ArrayLike,
    *,
    dimensions: int,
    field_name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    dimension_name = "one-dimensional" if dimensions == 1 else "two-dimensional"
    if array.ndim != dimensions:
        raise ValueError(f"{field_name} must be {dimension_name}")
    if array.size == 0 or any(size == 0 for size in array.shape):
        raise ValueError(f"{field_name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return array


def normalize_advantage_vector(
    advantages: ArrayLike,
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Normalize one vector for diagnostics with population statistics.

    This helper is observational only. The PPO actor uses the pinned Torch
    normalization after raw reward-cost composition.
    """

    threshold = _validated_epsilon(epsilon)
    vector = _finite_float_array(
        advantages,
        dimensions=1,
        field_name="advantages",
    )
    centered = vector - float(np.mean(vector))
    standard_deviation = float(np.std(vector, ddof=0))
    if standard_deviation <= threshold:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(centered / standard_deviation, dtype=np.float64)


def normalize_cost_advantages(
    cost_advantages: ArrayLike,
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Normalize cost columns independently for diagnostics only."""

    threshold = _validated_epsilon(epsilon)
    matrix = _finite_float_array(
        cost_advantages,
        dimensions=2,
        field_name="cost_advantages",
    )
    means = np.mean(matrix, axis=0)
    standard_deviations = np.std(matrix, axis=0, ddof=0)
    centered = matrix - means
    normalized = np.zeros_like(matrix, dtype=np.float64)
    active_columns = standard_deviations > threshold
    if np.any(active_columns):
        normalized[:, active_columns] = (
            centered[:, active_columns] / standard_deviations[active_columns]
        )
    return normalized


def combine_lagrangian_advantages(
    *,
    reward_advantages: ArrayLike,
    cost_advantages: ArrayLike,
    multipliers: ArrayLike,
) -> NDArray[np.float64]:
    """Return raw ``A_reward - sum(lambda_i * A_cost_i)`` in batch order."""

    reward_vector = _finite_float_array(
        reward_advantages,
        dimensions=1,
        field_name="reward_advantages",
    )
    cost_matrix = _finite_float_array(
        cost_advantages,
        dimensions=2,
        field_name="cost_advantages",
    )
    multiplier_vector = _finite_float_array(
        multipliers,
        dimensions=1,
        field_name="multipliers",
    )
    if cost_matrix.shape[0] != reward_vector.shape[0]:
        raise ValueError("reward and cost batch dimensions must match")
    if multiplier_vector.shape[0] != cost_matrix.shape[1]:
        raise ValueError("multipliers must contain one value per cost column")
    if np.any(multiplier_vector < 0.0):
        raise ValueError("multipliers must be non-negative")

    combined = reward_vector - cost_matrix @ multiplier_vector
    if not np.isfinite(combined).all():
        raise ValueError("combined Lagrangian advantages must be finite")
    return np.asarray(combined, dtype=np.float64)


__all__ = [
    "combine_lagrangian_advantages",
    "normalize_advantage_vector",
    "normalize_cost_advantages",
]
