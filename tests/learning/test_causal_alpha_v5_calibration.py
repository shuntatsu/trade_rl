from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
)
from trade_rl.learning.causal_alpha_v4 import build_causal_alpha_v4_forecast
from trade_rl.learning.causal_alpha_v5 import (
    CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV5CalibrationConfig,
    CausalAlphaV5CalibrationFit,
    V5SelectiveState,
    build_causal_alpha_v5_selective_forecast,
)


def _digest(char: str) -> str:
    return char * 64


def _ridge_model(
    *,
    intercept: float = 0.0,
    sample_count: int = 288,
) -> CausalAlphaRidgeModel:
    width = len(CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES)
    return CausalAlphaRidgeModel(
        feature_names=CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES,
        location=np.zeros(width, dtype=np.float64),
        scale=np.ones(width, dtype=np.float64),
        constant_mask=np.zeros(width, dtype=np.bool_),
        coefficients=np.zeros(width, dtype=np.float64),
        intercept=intercept,
        sample_count=sample_count,
        knowledge_cutoff=1_000,
        eligible_indices=np.arange(sample_count, dtype=np.int64),
        config=CausalAlphaRidgeConfig(ridge_strength=1.0),
    )


def _fit(
    *,
    intercept: float = 0.0,
    direction_score_rmse: float = 0.5,
) -> CausalAlphaV5CalibrationFit:
    return CausalAlphaV5CalibrationFit(
        v4_fit_digest=_digest("a"),
        v4_fit_config_digest=_digest("b"),
        v4_sample_scope_digest=_digest("c"),
        calibration_start=800,
        train_stop=1_000,
        model=_ridge_model(intercept=intercept),
        forward_model_digests=(_digest("d"), _digest("e"), _digest("f")),
        forward_residual_digests=(_digest("1"), _digest("2"), _digest("3")),
        final_weight_digest=_digest("4"),
        forward_weight_digests=(_digest("5"), _digest("6"), _digest("7")),
        per_symbol_support=(
            ("APTUSDT", 32),
            ("ARBUSDT", 32),
            ("BCHUSDT", 32),
            ("BNBUSDT", 32),
            ("BTCUSDT", 32),
            ("LINKUSDT", 32),
            ("LTCUSDT", 32),
            ("SOLUSDT", 32),
            ("XRPUSDT", 32),
        ),
        calibration_block_support=(72, 72, 72, 72),
        forward_block_symbol_counts=(9, 9, 9),
        calibration_residual_rmse=0.01,
        direction_score_rmse=direction_score_rmse,
        config=CausalAlphaV5CalibrationConfig(),
    )


def _forecast(
    *,
    rows: int = 3,
    direction_24h: float = 0.8,
    direction_72h: float = 0.6,
):
    market = {
        "4h": np.full(rows, 0.01),
        "24h": np.full(rows, 0.04),
        "72h": np.full(rows, 0.12),
    }
    residual = {horizon: np.zeros(rows) for horizon in market}
    direction = {
        "4h": np.full(rows, 1.0),
        "24h": np.full(rows, direction_24h),
        "72h": np.full(rows, direction_72h),
    }
    return build_causal_alpha_v4_forecast(
        symbol="ETHUSDT",
        decision_indices=np.arange(100, 100 + rows, dtype=np.int64),
        beta=np.ones(rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=direction,
        market_model_digests={horizon: _digest("8") for horizon in market},
        residual_model_digests={horizon: _digest("9") for horizon in market},
        direction_model_digests={horizon: _digest("0") for horizon in market},
        fit_digest=_digest("a"),
    )


def test_v5_calibration_config_is_frozen() -> None:
    config = CausalAlphaV5CalibrationConfig()

    assert config.calibration_fraction == 0.20
    assert config.forward_block_count == 4
    assert config.ridge_strength == 1.0
    assert config.minimum_pooled_support == 256
    assert config.minimum_symbol_support == 16
    assert config.minimum_selective_confidence == 1.0
    assert config.minimum_active_coverage == 0.25
    assert config.minimum_scope_active_fraction == 0.20
    assert config.minimum_scope_active_count == 3
    assert config.execution_cost_multiplier == 1.5
    assert config.edge_margin == 0.001


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("calibration_fraction", 0.25),
        ("forward_block_count", 5),
        ("ridge_strength", 0.1),
        ("minimum_pooled_support", 255),
        ("minimum_symbol_support", 15),
        ("minimum_selective_confidence", 0.9),
        ("minimum_active_coverage", 0.20),
        ("minimum_scope_active_fraction", 0.10),
        ("minimum_scope_active_count", 2),
        ("execution_cost_multiplier", 1.0),
        ("edge_margin", 0.0),
    ),
)
def test_v5_calibration_config_rejects_unapproved_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        CausalAlphaV5CalibrationConfig(**{field: value})


def test_v5_calibration_feature_contract_contains_no_symbol_identity() -> None:
    assert CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES == (
        "slow_return_raw",
        "slow_direction_raw",
        "log_slow_uncertainty",
        *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    )
    assert all(
        "symbol" not in name.lower()
        for name in CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES
    )
    assert all(
        "usdt" not in name.lower() for name in CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES
    )


def test_v5_calibration_fit_binds_support_and_model_identity() -> None:
    fit = _fit()

    assert fit.model.digest
    assert fit.pooled_support == 288
    assert fit.forward_model_digests == (_digest("d"), _digest("e"), _digest("f"))
    assert fit.digest


def test_v5_calibration_fit_rejects_model_support_mismatch() -> None:
    fit = _fit()

    with pytest.raises(ValueError, match="model sample_count"):
        CausalAlphaV5CalibrationFit(
            v4_fit_digest=fit.v4_fit_digest,
            v4_fit_config_digest=fit.v4_fit_config_digest,
            v4_sample_scope_digest=fit.v4_sample_scope_digest,
            calibration_start=fit.calibration_start,
            train_stop=fit.train_stop,
            model=_ridge_model(sample_count=287),
            forward_model_digests=fit.forward_model_digests,
            forward_residual_digests=fit.forward_residual_digests,
            final_weight_digest=fit.final_weight_digest,
            forward_weight_digests=fit.forward_weight_digests,
            per_symbol_support=fit.per_symbol_support,
            calibration_block_support=fit.calibration_block_support,
            forward_block_symbol_counts=fit.forward_block_symbol_counts,
            calibration_residual_rmse=fit.calibration_residual_rmse,
            direction_score_rmse=fit.direction_score_rmse,
            config=fit.config,
        )


def test_selective_forecast_has_no_direction_override_escape_hatch() -> None:
    parameters = inspect.signature(build_causal_alpha_v5_selective_forecast).parameters

    assert "slow_direction_override" not in parameters


def test_v5_calibration_fit_rejects_insufficient_symbol_support() -> None:
    fit = _fit()
    support = tuple(
        (symbol, 15 if symbol == "LTCUSDT" else count)
        for symbol, count in fit.per_symbol_support
    )

    with pytest.raises(ValueError, match="symbol support"):
        CausalAlphaV5CalibrationFit(
            v4_fit_digest=fit.v4_fit_digest,
            v4_fit_config_digest=fit.v4_fit_config_digest,
            v4_sample_scope_digest=fit.v4_sample_scope_digest,
            calibration_start=fit.calibration_start,
            train_stop=fit.train_stop,
            model=fit.model,
            forward_model_digests=fit.forward_model_digests,
            forward_residual_digests=fit.forward_residual_digests,
            final_weight_digest=fit.final_weight_digest,
            forward_weight_digests=fit.forward_weight_digests,
            per_symbol_support=support,
            calibration_block_support=fit.calibration_block_support,
            forward_block_symbol_counts=fit.forward_block_symbol_counts,
            calibration_residual_rmse=fit.calibration_residual_rmse,
            direction_score_rmse=fit.direction_score_rmse,
            config=fit.config,
        )


def test_selective_forecast_calibrates_slow_return_and_uncertainty() -> None:
    forecast = _forecast()
    fit = _fit(intercept=0.01)
    rows = len(forecast.decision_indices)
    selective = build_causal_alpha_v5_selective_forecast(
        v4_forecast=forecast,
        slow_uncertainty=np.full(rows, 0.005),
        instrument_descriptors=np.zeros(
            (rows, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)), dtype=np.float64
        ),
        instrument_descriptor_available=np.ones(
            (rows, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)), dtype=np.bool_
        ),
        one_way_cost_rates=np.zeros(rows),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        calibration_fit=fit,
    )

    expected_raw = 0.5 * (0.04 + 0.12 / 3.0)
    assert np.allclose(selective.slow_return_raw, expected_raw)
    assert np.allclose(selective.slow_direction_raw, 0.7)
    assert np.allclose(selective.slow_return_calibrated, expected_raw + 0.01)
    assert np.all(
        selective.slow_uncertainty_calibrated >= selective.slow_uncertainty_raw
    )
    assert np.all(selective.active_mask)
    assert set(selective.states) == {V5SelectiveState.ACTIVE}


def test_selective_forecast_confidence_equality_is_active() -> None:
    forecast = _forecast(rows=1)
    fit = _fit(direction_score_rmse=0.7)
    selective = build_causal_alpha_v5_selective_forecast(
        v4_forecast=forecast,
        slow_uncertainty=np.asarray([0.005]),
        instrument_descriptors=np.zeros((1, 9)),
        instrument_descriptor_available=np.ones((1, 9), dtype=np.bool_),
        one_way_cost_rates=np.asarray([0.0]),
        actionable_mask=np.asarray([True]),
        calibration_fit=fit,
    )

    assert selective.return_confidence[0] > 1.0
    assert math.isclose(selective.direction_confidence[0], 1.0)
    assert selective.selective_confidence[0] == 1.0
    assert selective.states[0] is V5SelectiveState.ACTIVE


def test_selective_forecast_hurdle_equality_abstains() -> None:
    forecast = _forecast(rows=1)
    fit = _fit()
    raw = 0.5 * (0.04 + 0.12 / 3.0)
    uncertainty = 0.01
    calibrated_uncertainty = math.sqrt(
        uncertainty**2 + fit.calibration_residual_rmse**2
    )
    cost = (
        raw - calibrated_uncertainty - fit.config.edge_margin
    ) / fit.config.execution_cost_multiplier
    selective = build_causal_alpha_v5_selective_forecast(
        v4_forecast=forecast,
        slow_uncertainty=np.asarray([uncertainty]),
        instrument_descriptors=np.zeros((1, 9)),
        instrument_descriptor_available=np.ones((1, 9), dtype=np.bool_),
        one_way_cost_rates=np.asarray([cost]),
        actionable_mask=np.asarray([True]),
        calibration_fit=fit,
    )

    assert np.isclose(
        abs(selective.slow_return_calibrated[0])
        - selective.slow_uncertainty_calibrated[0]
        - selective.execution_hurdle[0],
        0.0,
        atol=1e-12,
    )
    assert not bool(selective.active_mask[0])
    assert selective.states[0] is V5SelectiveState.EDGE_BELOW_HURDLE


def test_selective_forecast_blocks_disagreement_and_missing_descriptor() -> None:
    forecast = _forecast(
        rows=2,
        direction_24h=-1.0,
        direction_72h=-1.0,
    )
    fit = _fit()
    descriptors = np.zeros((2, 9))
    available = np.ones((2, 9), dtype=np.bool_)
    available[1, 0] = False
    selective = build_causal_alpha_v5_selective_forecast(
        v4_forecast=forecast,
        slow_uncertainty=np.full(2, 0.005),
        instrument_descriptors=descriptors,
        instrument_descriptor_available=available,
        one_way_cost_rates=np.zeros(2),
        actionable_mask=np.ones(2, dtype=np.bool_),
        calibration_fit=fit,
    )

    assert selective.states[0] is V5SelectiveState.DIRECTION_DISAGREEMENT
    assert selective.states[1] is V5SelectiveState.UNACTIONABLE
    assert not selective.active_mask.any()


@pytest.mark.parametrize(
    "field",
    ("forward_block_count", "minimum_pooled_support", "minimum_symbol_support"),
)
def test_v5_calibration_config_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(ValueError):
        CausalAlphaV5CalibrationConfig(**{field: True})


def test_v5_example_json_freezes_task1_hypothesis() -> None:
    payload = json.loads(
        Path("examples/binance/universal-causal-alpha-v5-research.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload == {
        "schema_version": "universal_causal_alpha_v5_research_config_v1",
        "calibration": {
            "calibration_fraction": 0.2,
            "forward_block_count": 4,
            "ridge_strength": 1.0,
            "minimum_pooled_support": 256,
            "minimum_symbol_support": 16,
            "minimum_selective_confidence": 1.0,
            "minimum_active_coverage": 0.25,
            "minimum_scope_active_fraction": 0.2,
            "minimum_scope_active_count": 3,
            "execution_cost_multiplier": 1.5,
            "edge_margin": 0.001,
            "epsilon": 1e-12,
        },
    }


def test_learning_package_exports_v5_contracts() -> None:
    import trade_rl.learning as learning

    for name in (
        "CAUSAL_ALPHA_V5_CALIBRATION_FEATURE_NAMES",
        "CausalAlphaV5CalibrationConfig",
        "CausalAlphaV5CalibrationFit",
        "CausalAlphaV5SelectiveForecast",
        "V5SelectiveState",
        "build_causal_alpha_v5_selective_forecast",
    ):
        assert getattr(learning, name) is not None
