from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    fit_causal_alpha_ridge,
)


def _fit(available: np.ndarray):
    features = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=np.float64,
    )
    labels = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    return fit_causal_alpha_ridge(
        features=features,
        labels=labels,
        feature_available=available,
        label_end_indices=np.asarray([1, 2, 3, 4], dtype=np.int64),
        knowledge_cutoff=5,
        feature_names=("fast", "slow"),
        config=CausalAlphaRidgeConfig(ridge_strength=0.1),
    )


def test_missing_feature_does_not_discard_otherwise_eligible_label_row() -> None:
    available = np.asarray(
        [[True, True], [True, False], [True, True], [True, True]],
        dtype=np.bool_,
    )
    model = _fit(available)

    assert model.sample_count == 4
    assert model.eligible_indices.tolist() == [0, 1, 2, 3]
    assert model.location[0] == pytest.approx(2.5)
    assert model.location[1] == pytest.approx((10.0 + 30.0 + 40.0) / 3.0)

    transformed = model.transform(
        np.asarray([[2.0, 20.0]], dtype=np.float64),
        feature_available=np.asarray([[True, False]], dtype=np.bool_),
    )
    assert transformed[0, 1] == pytest.approx(0.0)


def test_never_available_feature_becomes_deterministic_zero_column() -> None:
    available = np.asarray(
        [[True, False], [True, False], [True, False], [True, False]],
        dtype=np.bool_,
    )
    model = _fit(available)

    assert model.sample_count == 4
    assert model.location[1] == pytest.approx(0.0)
    assert model.scale[1] == pytest.approx(1.0)
    assert bool(model.constant_mask[1]) is True
    assert model.coefficients[1] == pytest.approx(0.0)
