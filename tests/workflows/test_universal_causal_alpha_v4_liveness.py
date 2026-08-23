from __future__ import annotations

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    build_causal_alpha_v4_liveness_evidence,
)


def _digest(char: str) -> str:
    return char * 64


def _contributions(rows: int) -> dict[str, np.ndarray]:
    index = np.arange(rows, dtype=np.float64)
    return {
        "existing_15m": 0.001 * np.sin(index),
        "existing_1h": 0.002 * np.cos(index / 2.0),
        "existing_4h": 0.003 * np.sin(index / 3.0),
        "existing_1d": 0.004 * np.cos(index / 4.0),
        "local_cross_market": 0.005 * np.sin(index / 5.0),
        "global_market": 0.006 * np.cos(index / 6.0),
        "beta_scaled_proxy": 0.007 * np.sin(index / 7.0),
        "shared_residual": 0.008 * np.cos(index / 8.0),
    }


def test_v4_liveness_reports_prediction_runs_and_contribution_variance() -> None:
    prediction = np.asarray([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float64)
    direction = np.asarray([1.0, 1.0, -1.0, 0.5, -0.5, 1.0], dtype=np.float64)
    available = np.ones((len(prediction), 4), dtype=np.bool_)
    constant_mask = np.asarray([False, False, True, False], dtype=np.bool_)
    contributions = _contributions(len(prediction))

    evidence = build_causal_alpha_v4_liveness_evidence(
        fit_digest=_digest("a"),
        forecast_digest=_digest("b"),
        symbol="ETHUSDT",
        horizon="4h",
        prediction=prediction,
        direction_score=direction,
        intercept=0.05,
        weighted_final_rmse=0.025,
        feature_available=available,
        constant_feature_mask=constant_mask,
        contribution_series=contributions,
    )

    assert evidence.unique_count_at_tolerance_1e_12 == 3
    assert evidence.median_near_identical_run_length == 2.0
    assert evidence.maximum_near_identical_run_length == 3
    assert np.isclose(evidence.dynamic_prediction_std, np.std(prediction - 0.05))
    assert np.isclose(
        evidence.contribution_variance_local_cross_market,
        np.var(contributions["local_cross_market"]),
    )
    assert evidence.constant_feature_count == 1
    assert evidence.available_feature_count == 4
    assert evidence.direction_positive_fraction == 4.0 / 6.0
    assert evidence.direction_negative_fraction == 2.0 / 6.0
    assert evidence.research_only is True
    assert evidence.promotion_eligible is False


def test_v4_liveness_rejects_intercept_only_prediction_with_live_features() -> None:
    rows = 8
    with pytest.raises(ValueError, match="dynamic prediction"):
        build_causal_alpha_v4_liveness_evidence(
            fit_digest=_digest("c"),
            forecast_digest=_digest("d"),
            symbol="BTCUSDT",
            horizon="24h",
            prediction=np.full(rows, 0.003, dtype=np.float64),
            direction_score=np.ones(rows, dtype=np.float64),
            intercept=0.003,
            weighted_final_rmse=0.01,
            feature_available=np.ones((rows, 3), dtype=np.bool_),
            constant_feature_mask=np.asarray([False, False, True], dtype=np.bool_),
            contribution_series={
                key: np.zeros(rows, dtype=np.float64) for key in _contributions(rows)
            },
        )


def test_v4_liveness_allows_constant_prediction_when_no_dynamic_feature_support() -> None:
    rows = 8
    evidence = build_causal_alpha_v4_liveness_evidence(
        fit_digest=_digest("e"),
        forecast_digest=_digest("f"),
        symbol="BTCUSDT",
        horizon="72h",
        prediction=np.full(rows, 0.003, dtype=np.float64),
        direction_score=np.zeros(rows, dtype=np.float64),
        intercept=0.003,
        weighted_final_rmse=0.01,
        feature_available=np.ones((rows, 2), dtype=np.bool_),
        constant_feature_mask=np.ones(2, dtype=np.bool_),
        contribution_series={
            key: np.zeros(rows, dtype=np.float64) for key in _contributions(rows)
        },
    )

    assert evidence.dynamic_prediction_std == 0.0
    assert evidence.constant_feature_count == 2
    assert evidence.available_feature_count == 2
