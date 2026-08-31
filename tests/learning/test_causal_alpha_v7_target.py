from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4Forecast,
    build_causal_alpha_v4_forecast,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetConfig,
)
from trade_rl.learning.causal_alpha_v6_target import causal_alpha_v6_target_path
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
    CausalAlphaV7Candidate,
)
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationFit,
    CausalAlphaV7CalibrationRows,
    fit_causal_alpha_v7_calibration,
)
from trade_rl.learning.causal_alpha_v7_target import causal_alpha_v7_target_paths


def _digest(char: str) -> str:
    return char * 64


def _range() -> CausalAlphaV7CalibrationRange:
    return CausalAlphaV7CalibrationRange(
        base_fit_cutoff=500,
        calibration_start=500,
        train_stop=1_000,
        block_boundaries=(500, 625, 750, 875, 1_000),
        split_digest=_digest("a"),
        feature_names=CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    )


def _features(count: int, phase: float) -> np.ndarray:
    x = np.arange(count, dtype=np.float64)
    fast = 0.02 * np.sin(x * 0.071 + phase)
    direction = np.sin(x * 0.053 + phase)
    return np.column_stack(
        (
            fast,
            direction,
            np.full(count, np.log(0.01)),
            0.01 + 0.002 * np.cos(x * 0.011),
            1.0 + 0.1 * np.sin(x * 0.017),
            0.2 + 0.05 * np.cos(x * 0.019),
            2.0 * fast,
            3.0 * fast,
            0.5 * direction,
            0.25 * direction,
        )
    )


def _calibration_fit() -> CausalAlphaV7CalibrationFit:
    rows: dict[str, CausalAlphaV7CalibrationRows] = {}
    for symbol, phase in (("BTCUSDT", 0.3), ("SOLUSDT", 0.9)):
        count = 160
        decisions = np.arange(500, 500 + count, dtype=np.int64)
        features = _features(count, phase)
        realized = 0.006 * np.sin(decisions * 0.083 + phase) + 0.25 * features[:, 0]
        rows[symbol] = CausalAlphaV7CalibrationRows(
            symbol=symbol,
            decision_indices=decisions,
            label_end_indices=decisions + 1,
            features=features,
            feature_available=np.ones(features.shape, dtype=np.bool_),
            realized_returns=realized,
            range_digest=_range().digest,
        )
    return fit_causal_alpha_v7_calibration(
        rows=rows,
        calibration_range=_range(),
        config=CausalAlphaV7CalibrationConfig(),
    )


def _forecast(rows: int = 12) -> CausalAlphaV4Forecast:
    decisions = np.arange(1_000, 1_000 + rows, dtype=np.int64)
    fast = np.linspace(-0.03, 0.03, rows)
    return build_causal_alpha_v4_forecast(
        symbol="BTCUSDT",
        decision_indices=decisions,
        beta=np.ones(rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions={
            "4h": 0.4 * fast,
            "24h": np.full(rows, 0.01),
            "72h": np.full(rows, 0.02),
        },
        residual_predictions={
            "4h": 0.6 * fast,
            "24h": np.full(rows, 0.005),
            "72h": np.full(rows, 0.01),
        },
        direction_scores={
            "4h": np.sign(fast),
            "24h": np.ones(rows),
            "72h": np.ones(rows),
        },
        market_model_digests={
            horizon: _digest("b") for horizon in ("4h", "24h", "72h")
        },
        residual_model_digests={
            horizon: _digest("c") for horizon in ("4h", "24h", "72h")
        },
        direction_model_digests={
            horizon: _digest("d") for horizon in ("4h", "24h", "72h")
        },
        fit_digest=_digest("e"),
    )


def test_v7_control_is_exactly_the_v6_fast_only_target_path() -> None:
    forecast = _forecast()
    rows = len(forecast.decision_indices)
    uncertainty = {horizon: np.full(rows, 0.001) for horizon in ("4h", "24h", "72h")}
    costs = np.full(rows, 0.0001)
    caps = np.full(rows, 0.25)
    actionable = np.ones(rows, dtype=np.bool_)
    config = CausalAlphaV6TargetConfig()

    paths = causal_alpha_v7_target_paths(
        forecast=forecast,
        calibration_fit=_calibration_fit(),
        calibration_features=_features(rows, 0.2),
        calibration_feature_available=np.ones((rows, 10), dtype=np.bool_),
        uncertainty=uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
        actionable_mask=actionable,
        config=config,
        initial_weight=0.0,
    )
    direct = causal_alpha_v6_target_path(
        forecast,
        uncertainty=uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
        actionable_mask=actionable,
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        config=config,
        initial_weight=0.0,
    )

    assert tuple(paths) == tuple(CausalAlphaV7Candidate)
    assert (
        paths[CausalAlphaV7Candidate.V6_CONTROL].v6_target_path.digest == direct.digest
    )


def test_v7_candidates_change_only_fast_forecast_inputs() -> None:
    forecast = _forecast()
    rows = len(forecast.decision_indices)
    calibration_fit = _calibration_fit()
    features = _features(rows, 0.2)
    available = np.ones(features.shape, dtype=np.bool_)
    expected_calibrated, expected_reliability = calibration_fit.predict(
        features,
        feature_available=available,
    )
    costs = np.full(rows, 0.0001)
    caps = np.linspace(0.10, 0.25, rows)
    risk_caps = np.linspace(0.25, 0.15, rows)
    uncertainty = {horizon: np.full(rows, 0.001) for horizon in ("4h", "24h", "72h")}

    paths = causal_alpha_v7_target_paths(
        forecast=forecast,
        calibration_fit=calibration_fit,
        calibration_features=features,
        calibration_feature_available=available,
        uncertainty=uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
        risk_weight_caps=risk_caps,
        actionable_mask=np.ones(rows, dtype=np.bool_),
        config=CausalAlphaV6TargetConfig(),
        initial_weight=0.0,
    )
    control = paths[CausalAlphaV7Candidate.V6_CONTROL].v6_target_path
    contrarian = paths[CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN].v6_target_path
    calibrated = paths[CausalAlphaV7Candidate.CAUSAL_CALIBRATED].v6_target_path

    np.testing.assert_allclose(
        contrarian.expected_returns_4h,
        -control.expected_returns_4h,
    )
    np.testing.assert_allclose(
        contrarian.direction_scores_4h,
        -control.direction_scores_4h,
    )
    np.testing.assert_allclose(calibrated.expected_returns_4h, expected_calibrated)
    np.testing.assert_allclose(calibrated.direction_scores_4h, expected_reliability)
    for path in (control, contrarian, calibrated):
        np.testing.assert_array_equal(path.uncertainties_4h, uncertainty["4h"])
        np.testing.assert_array_equal(path.one_way_cost_rates, costs)
        np.testing.assert_array_equal(path.liquidity_weight_caps, caps)
        np.testing.assert_array_equal(path.risk_weight_caps, risk_caps)
        assert path.config_digest == CausalAlphaV6TargetConfig().digest
