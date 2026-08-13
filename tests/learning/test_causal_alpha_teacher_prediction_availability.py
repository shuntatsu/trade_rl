from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    causal_alpha_target_path,
    fit_causal_alpha_ridge,
)


def _model():
    features = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=np.float64,
    )
    return fit_causal_alpha_ridge(
        features=features,
        labels=2.0 * features[:, 0] + 0.1 * features[:, 1],
        feature_available=np.ones_like(features, dtype=np.bool_),
        label_end_indices=np.asarray([1, 2, 3, 4], dtype=np.int64),
        knowledge_cutoff=5,
        feature_names=("fast", "slow"),
        config=CausalAlphaRidgeConfig(ridge_strength=0.1),
    )


def test_prediction_unavailable_features_use_fitted_mean_semantics() -> None:
    model = _model()
    raw = np.asarray([[9999.0, 35.0]], dtype=np.float64)
    availability = np.asarray([[False, True]], dtype=np.bool_)

    actual = model.predict(raw, feature_available=availability)
    expected = model.predict(
        np.asarray([[model.location[0], 35.0]], dtype=np.float64)
    )

    assert actual.tolist() == pytest.approx(expected.tolist())


def test_prediction_availability_shape_is_fail_closed() -> None:
    model = _model()
    with pytest.raises(ValueError, match="availability"):
        model.predict(
            np.asarray([[1.0, 2.0]], dtype=np.float64),
            feature_available=np.asarray([True], dtype=np.bool_),
        )


def test_controller_holds_target_on_non_actionable_decision() -> None:
    from trade_rl.learning.causal_alpha_teacher import (
        CausalAlphaControllerConfig,
        CausalAlphaHorizonMix,
    )

    path = causal_alpha_target_path(
        np.asarray([0.5, -0.5, -0.5], dtype=np.float64),
        config=CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.H24,
            score_scale=10.0,
            entry_threshold=0.05,
            exit_threshold=0.01,
            no_trade_band=0.0,
            max_target_delta=2.0,
        ),
        initial_weight=0.0,
        actionable_mask=np.asarray([True, False, True], dtype=np.bool_),
    )

    assert path.targets[0] > 0.0
    assert path.targets[1] == pytest.approx(path.targets[0])
    assert path.targets[2] < 0.0
    assert path.actionable_mask.tolist() == [True, False, True]
