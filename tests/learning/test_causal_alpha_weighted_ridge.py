from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    fit_causal_alpha_ridge,
)


def _fit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    normalize_objective: bool = False,
    working_memory_rows: int | None = None,
):
    return fit_causal_alpha_ridge(
        features=features,
        labels=labels,
        feature_available=np.ones_like(features, dtype=np.bool_),
        label_end_indices=np.arange(1, len(labels) + 1, dtype=np.int64),
        knowledge_cutoff=len(labels) + 2,
        feature_names=tuple(f"x{index}" for index in range(features.shape[1])),
        config=CausalAlphaRidgeConfig(ridge_strength=1e-9),
        sample_weights=sample_weights,
        normalize_objective=normalize_objective,
        working_memory_rows=working_memory_rows,
    )


def test_default_weight_arguments_preserve_legacy_model_identity() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    common = dict(
        features=features,
        labels=labels,
        feature_available=np.ones_like(features, dtype=np.bool_),
        label_end_indices=np.asarray([1, 2, 3, 4], dtype=np.int64),
        knowledge_cutoff=5,
        feature_names=("signal",),
        config=CausalAlphaRidgeConfig(ridge_strength=0.25),
    )

    legacy = fit_causal_alpha_ridge(**common)
    explicit = fit_causal_alpha_ridge(
        **common,
        sample_weights=None,
        normalize_objective=False,
    )

    assert explicit.to_payload() == legacy.to_payload()
    assert explicit.digest == legacy.digest
    assert explicit.predict(features).tolist() == legacy.predict(features).tolist()


def test_weighted_ridge_downweights_a_large_outlier() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([0.0, 1.0, 2.0, 30.0], dtype=np.float64)

    unweighted = _fit(features, labels)
    weighted = _fit(
        features,
        labels,
        sample_weights=np.asarray([1.0, 1.0, 1.0, 0.001], dtype=np.float64),
    )

    assert abs(weighted.coefficients[0] - 1.0) < abs(unweighted.coefficients[0] - 1.0)
    assert (
        weighted.predict(np.asarray([[4.0]])).item()
        < unweighted.predict(np.asarray([[4.0]])).item()
    )


@pytest.mark.parametrize(
    "weights",
    (
        np.asarray([1.0, -1.0, 1.0, 1.0]),
        np.asarray([1.0, np.nan, 1.0, 1.0]),
        np.asarray([0.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 1.0]),
    ),
)
def test_weighted_ridge_rejects_invalid_sample_weights(weights: np.ndarray) -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="sample_weights"):
        _fit(features, labels, sample_weights=weights)


def test_mean_objective_normalization_is_invariant_to_full_row_duplication() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    first = _fit(
        features,
        labels,
        sample_weights=np.ones(4, dtype=np.float64),
        normalize_objective=True,
    )
    second = _fit(
        np.repeat(features, 2, axis=0),
        np.repeat(labels, 2),
        sample_weights=np.ones(8, dtype=np.float64),
        normalize_objective=True,
    )

    assert second.intercept == pytest.approx(first.intercept, abs=1e-12)
    assert second.coefficients == pytest.approx(first.coefficients, abs=1e-12)


def test_chunked_weighted_ridge_matches_dense_weighted_objective() -> None:
    features = np.asarray(
        [[0.0, 2.0], [1.0, 0.0], [2.0, 3.0], [4.0, 1.0], [5.0, 6.0]],
        dtype=np.float64,
    )
    labels = np.asarray([1.0, -1.0, 2.0, 3.0, 8.0], dtype=np.float64)
    weights = np.asarray([0.1, 0.2, 0.3, 0.15, 0.25], dtype=np.float64)

    dense = _fit(
        features,
        labels,
        sample_weights=weights,
        normalize_objective=True,
    )
    chunked = _fit(
        features,
        labels,
        sample_weights=weights,
        normalize_objective=True,
        working_memory_rows=2,
    )

    np.testing.assert_array_equal(chunked.eligible_indices, dense.eligible_indices)
    np.testing.assert_allclose(chunked.location, dense.location, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(chunked.scale, dense.scale, atol=1e-14, rtol=0.0)
    assert chunked.intercept == pytest.approx(dense.intercept, abs=1e-12)
    np.testing.assert_allclose(
        chunked.coefficients, dense.coefficients, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        chunked.predict(features), dense.predict(features), atol=1e-12, rtol=0.0
    )
