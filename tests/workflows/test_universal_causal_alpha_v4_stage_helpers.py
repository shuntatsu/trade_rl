from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_teacher import CausalAlphaRidgeConfig, fit_causal_alpha_ridge
from trade_rl.learning.causal_alpha_v4 import build_causal_alpha_v4_forecast
from trade_rl.workflows.universal_causal_alpha_v4_liveness_inputs import (
    build_causal_alpha_v4_liveness_inputs,
)
from trade_rl.workflows.universal_causal_alpha_v4_stage_runner import (
    slice_causal_alpha_v4_forecast,
)


def _digest(char: str) -> str:
    return char * 64


def _forecast():
    rows = 6
    market = {
        "4h": np.linspace(0.01, 0.06, rows),
        "24h": np.linspace(0.02, 0.12, rows),
        "72h": np.linspace(0.06, 0.36, rows),
    }
    residual = {h: np.linspace(0.001, 0.006, rows) for h in market}
    direction = {h: np.linspace(-1.0, 1.0, rows) for h in market}
    return build_causal_alpha_v4_forecast(
        symbol="ETHUSDT",
        decision_indices=np.arange(100, 106, dtype=np.int64),
        beta=np.linspace(0.5, 1.0, rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=direction,
        market_model_digests={h: _digest("1") for h in market},
        residual_model_digests={h: _digest("2") for h in market},
        direction_model_digests={h: _digest("3") for h in market},
        fit_digest=_digest("4"),
    )


def test_v4_forecast_slice_keeps_every_component_aligned() -> None:
    source = _forecast()
    sliced = slice_causal_alpha_v4_forecast(source, np.asarray([1, 3, 4], dtype=np.int64))

    np.testing.assert_array_equal(sliced.decision_indices, np.asarray([101, 103, 104]))
    np.testing.assert_allclose(sliced.beta, source.beta[[1, 3, 4]])
    for horizon in ("4h", "24h", "72h"):
        np.testing.assert_allclose(
            sliced.market_predictions[horizon], source.market_predictions[horizon][[1, 3, 4]]
        )
        np.testing.assert_allclose(
            sliced.residual_predictions[horizon], source.residual_predictions[horizon][[1, 3, 4]]
        )
        np.testing.assert_allclose(
            sliced.direction_scores[horizon], source.direction_scores[horizon][[1, 3, 4]]
        )
        np.testing.assert_allclose(
            sliced.final_predictions[horizon], source.final_predictions[horizon][[1, 3, 4]]
        )


def test_v4_liveness_inputs_attribute_shared_linear_families_without_forging_proxy() -> None:
    rows = 8
    names = (
        "15m__a",
        "1h__b",
        "4h__c",
        "1d__d",
        "local_x",
        "global_x",
        "fee_rate",
        "causal_beta",
    )
    x = np.column_stack(
        [np.linspace(float(i), float(i + 1), rows) for i in range(len(names))]
    )
    available = np.ones_like(x, dtype=np.bool_)
    labels = 0.01 + x @ np.linspace(0.001, 0.008, len(names))
    model = fit_causal_alpha_ridge(
        features=x,
        labels=labels,
        feature_available=available,
        label_end_indices=np.arange(rows, dtype=np.int64),
        knowledge_cutoff=rows + 1,
        feature_names=names,
        config=CausalAlphaRidgeConfig(ridge_strength=0.1),
    )
    forecast = _forecast()
    forecast = slice_causal_alpha_v4_forecast(forecast, np.arange(rows - 2, dtype=np.int64))
    sample = SimpleNamespace(
        target_local_feature_names=names[:4],
        target_local_features=x[: rows - 2, :4],
        target_local_available=available[: rows - 2, :4],
        local_context=SimpleNamespace(
            feature_names=("local_x",),
            values=x[: rows - 2, 4:5],
            available=available[: rows - 2, 4:5],
        ),
        global_context=SimpleNamespace(
            feature_names=("global_x",),
            values=x[: rows - 2, 5:6],
            available=available[: rows - 2, 5:6],
        ),
        instrument_descriptor_names=("fee_rate",),
        instrument_descriptors=x[: rows - 2, 6:7],
        instrument_descriptor_available=available[: rows - 2, 6:7],
        beta=x[: rows - 2, 7],
        beta_available=np.ones(rows - 2, dtype=np.bool_),
    )
    fit = SimpleNamespace(residual_models={"4h": model})

    result = build_causal_alpha_v4_liveness_inputs(
        fit=fit,
        sample=sample,
        forecast=forecast,
        horizon="4h",
    )

    assert tuple(result.contribution_series) == (
        "existing_15m",
        "existing_1h",
        "existing_4h",
        "existing_1d",
        "local_cross_market",
        "global_market",
        "beta_scaled_proxy",
        "shared_residual",
    )
    np.testing.assert_allclose(
        result.contribution_series["beta_scaled_proxy"],
        forecast.beta_scaled_market_contributions["4h"],
    )
    np.testing.assert_allclose(
        result.contribution_series["shared_residual"], forecast.residual_predictions["4h"]
    )
    assert result.feature_available.shape[0] == rows - 2
    assert result.constant_feature_mask.shape == (len(names),)
