from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4FitConfig,
    build_causal_alpha_v4_forecast,
)


def _digest(char: str) -> str:
    return char * 64


def test_v4_fit_config_is_the_single_authored_hypothesis() -> None:
    config = CausalAlphaV4FitConfig(
        market_ridge_strength=1.0,
        residual_ridge_strength=0.1,
        direction_ridge_strength=0.1,
    )
    assert len(config.digest) == 64

    with pytest.raises(ValueError, match="market"):
        CausalAlphaV4FitConfig(
            market_ridge_strength=0.5,
            residual_ridge_strength=0.1,
            direction_ridge_strength=0.1,
        )
    with pytest.raises(ValueError, match="residual"):
        CausalAlphaV4FitConfig(
            market_ridge_strength=1.0,
            residual_ridge_strength=1.0,
            direction_ridge_strength=0.1,
        )
    with pytest.raises(ValueError, match="direction"):
        CausalAlphaV4FitConfig(
            market_ridge_strength=1.0,
            residual_ridge_strength=0.1,
            direction_ridge_strength=1.0,
        )


def test_v4_forecast_composes_beta_scaled_market_and_shared_residual() -> None:
    market = {
        "4h": np.asarray([0.01, -0.02]),
        "24h": np.asarray([0.03, 0.04]),
        "72h": np.asarray([0.09, -0.06]),
    }
    residual = {
        "4h": np.asarray([0.005, 0.004]),
        "24h": np.asarray([-0.002, 0.001]),
        "72h": np.asarray([0.003, -0.004]),
    }
    direction = {
        "4h": np.asarray([0.8, -0.4]),
        "24h": np.asarray([0.3, 0.6]),
        "72h": np.asarray([0.7, -0.2]),
    }
    beta = np.asarray([1.5, 0.5])

    forecast = build_causal_alpha_v4_forecast(
        symbol="ETHUSDT",
        decision_indices=np.asarray([100, 101], dtype=np.int64),
        beta=beta,
        beta_available=np.asarray([True, True], dtype=np.bool_),
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=direction,
        market_model_digests={horizon: _digest("1") for horizon in market},
        residual_model_digests={horizon: _digest("2") for horizon in market},
        direction_model_digests={horizon: _digest("3") for horizon in market},
        fit_digest=_digest("4"),
    )

    for horizon in ("4h", "24h", "72h"):
        expected_market = beta * market[horizon]
        np.testing.assert_allclose(
            forecast.beta_scaled_market_contributions[horizon],
            expected_market,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            forecast.final_predictions[horizon],
            expected_market + residual[horizon],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            forecast.direction_scores[horizon], direction[horizon]
        )
    assert forecast.symbol == "ETHUSDT"
    assert len(forecast.digest) == 64


def test_v4_forecast_rejects_horizon_or_shape_drift() -> None:
    common = dict(
        symbol="ETHUSDT",
        decision_indices=np.asarray([100, 101], dtype=np.int64),
        beta=np.asarray([1.0, 1.0]),
        beta_available=np.asarray([True, True], dtype=np.bool_),
        residual_predictions={
            "4h": np.zeros(2),
            "24h": np.zeros(2),
            "72h": np.zeros(2),
        },
        direction_scores={
            "4h": np.ones(2),
            "24h": np.ones(2),
            "72h": np.ones(2),
        },
        market_model_digests={
            horizon: _digest("1") for horizon in ("4h", "24h", "72h")
        },
        residual_model_digests={
            horizon: _digest("2") for horizon in ("4h", "24h", "72h")
        },
        direction_model_digests={
            horizon: _digest("3") for horizon in ("4h", "24h", "72h")
        },
        fit_digest=_digest("4"),
    )
    with pytest.raises(ValueError, match="horizon"):
        build_causal_alpha_v4_forecast(
            market_predictions={"4h": np.zeros(2), "24h": np.zeros(2)},
            **common,
        )
    with pytest.raises(ValueError, match="aligned|shape"):
        build_causal_alpha_v4_forecast(
            market_predictions={
                "4h": np.zeros(2),
                "24h": np.zeros(3),
                "72h": np.zeros(2),
            },
            **common,
        )
