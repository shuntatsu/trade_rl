"""Numeric-only weighted Ridge arithmetic shared across distinct return units."""

from __future__ import annotations

import numpy as np

_SCALE_FLOOR = 1e-12


def _solve_weighted_ridge(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Preserve legacy operation order; callers own row/unit validation."""
    weight_sum = float(weights.sum())
    feature_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    centered_x = x - feature_mean
    weighted_variance = (
        np.sum(
            centered_x**2 * weights[:, None],
            axis=0,
        )
        / weight_sum
    )
    raw_scale = np.sqrt(weighted_variance)
    feature_scale = np.where(raw_scale > _SCALE_FLOOR, raw_scale, 1.0)
    standardized = centered_x / feature_scale

    intercept = float(np.dot(weights, y) / weight_sum)
    centered_y = y - intercept
    sqrt_weights = np.sqrt(weights)
    weighted_x = standardized * sqrt_weights[:, None]
    weighted_y = centered_y * sqrt_weights
    gram = weighted_x.T @ weighted_x
    regularized = gram + alpha * np.eye(x.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(regularized, weighted_x.T @ weighted_y)
    return feature_mean, feature_scale, coefficients, intercept
