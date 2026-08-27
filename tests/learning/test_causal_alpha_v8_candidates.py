from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v4 import build_causal_alpha_v4_forecast
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v6_target import causal_alpha_v6_target_path
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
)
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationRows,
    fit_causal_alpha_v7_calibration,
)
from trade_rl.learning.causal_alpha_v7_target import causal_alpha_v7_target_paths
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
)
from trade_rl.learning.causal_alpha_v8_target import (
    causal_alpha_v8_target_paths,
    causal_alpha_v8_target_paths_from_v7,
)


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


def _calibration_fit() -> object:
    records: dict[str, CausalAlphaV7CalibrationRows] = {}
    for symbol, phase in (("BTCUSDT", 0.3), ("SOLUSDT", 0.9)):
        count = 160
        decisions = np.arange(500, 500 + count, dtype=np.int64)
        features = _features(count, phase)
        records[symbol] = CausalAlphaV7CalibrationRows(
            symbol=symbol,
            decision_indices=decisions,
            label_end_indices=decisions + 1,
            features=features,
            feature_available=np.ones_like(features, dtype=np.bool_),
            realized_returns=0.006 * np.sin(decisions * 0.083 + phase)
            + 0.25 * features[:, 0],
            range_digest=_range().digest,
        )
    return fit_causal_alpha_v7_calibration(
        rows=records,
        calibration_range=_range(),
        config=CausalAlphaV7CalibrationConfig(),
    )


def _forecast(rows: int = 12) -> object:
    fast = np.linspace(-0.03, 0.03, rows)
    zeros = np.zeros(rows)
    return build_causal_alpha_v4_forecast(
        symbol="BTCUSDT",
        decision_indices=np.arange(1_000, 1_000 + rows),
        beta=np.ones(rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions={"4h": 0.4 * fast, "24h": zeros, "72h": zeros},
        residual_predictions={"4h": 0.6 * fast, "24h": zeros, "72h": zeros},
        direction_scores={
            "4h": np.sign(fast),
            "24h": zeros,
            "72h": zeros,
        },
        market_model_digests={h: _digest("b") for h in ("4h", "24h", "72h")},
        residual_model_digests={h: _digest("c") for h in ("4h", "24h", "72h")},
        direction_model_digests={h: _digest("d") for h in ("4h", "24h", "72h")},
        fit_digest=_digest("e"),
    )


def _inputs() -> dict[str, object]:
    forecast = _forecast()
    rows = len(forecast.decision_indices)  # type: ignore[attr-defined]
    return {
        "forecast": forecast,
        "calibration_fit": _calibration_fit(),
        "calibration_features": _features(rows, 0.2),
        "calibration_feature_available": np.ones((rows, 10), dtype=np.bool_),
        "uncertainty": {h: np.full(rows, 0.001) for h in ("4h", "24h", "72h")},
        "one_way_cost_rates": np.full(rows, 0.0001),
        "liquidity_weight_caps": np.full(rows, 0.25),
        "risk_weight_caps": np.full(rows, 0.25),
        "actionable_mask": np.ones(rows, dtype=np.bool_),
        "config": CausalAlphaV8TargetConfig(),
        "initial_weight": 0.0,
    }


def test_v8_candidates_are_fixed_and_control_is_exact_v7_path() -> None:
    inputs = _inputs()
    paths = causal_alpha_v8_target_paths(**inputs)
    forecast = inputs["forecast"]
    config = inputs["config"]
    assert isinstance(config, CausalAlphaV8TargetConfig)
    direct = causal_alpha_v6_target_path(
        forecast,  # type: ignore[arg-type]
        uncertainty=inputs["uncertainty"],  # type: ignore[arg-type]
        one_way_cost_rates=inputs["one_way_cost_rates"],
        liquidity_weight_caps=inputs["liquidity_weight_caps"],
        risk_weight_caps=inputs["risk_weight_caps"],
        actionable_mask=inputs["actionable_mask"],
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        config=config.base,
        initial_weight=0.0,
    )

    assert tuple(paths) == tuple(CausalAlphaV8Candidate)
    assert (
        paths[CausalAlphaV8Candidate.V7_CONTROL].v6_target_path.digest == direct.digest
    )


def test_v8_robust_candidates_bind_expected_fast_transformations() -> None:
    paths = causal_alpha_v8_target_paths(**_inputs())
    control = paths[CausalAlphaV8Candidate.V7_CONTROL].v6_target_path
    contrarian = paths[CausalAlphaV8Candidate.ROBUST_CONTRARIAN].v6_target_path
    calibrated = paths[CausalAlphaV8Candidate.ROBUST_CALIBRATED].v6_target_path

    np.testing.assert_allclose(
        contrarian.expected_returns_4h, -control.expected_returns_4h
    )
    np.testing.assert_allclose(
        contrarian.direction_scores_4h, -control.direction_scores_4h
    )
    assert contrarian.config_digest == CausalAlphaV8TargetConfig().digest
    assert calibrated.config_digest == CausalAlphaV8TargetConfig().digest
    assert contrarian.sign_flip_count == 0
    assert calibrated.sign_flip_count == 0


def test_v8_can_recompile_frozen_v7_effective_forecasts() -> None:
    inputs = _inputs()
    config = inputs["config"]
    assert isinstance(config, CausalAlphaV8TargetConfig)
    v7_inputs = {key: value for key, value in inputs.items() if key != "config"}
    v7_paths = causal_alpha_v7_target_paths(**v7_inputs, config=config.base)  # type: ignore[arg-type]

    paths = causal_alpha_v8_target_paths_from_v7(
        forecast=inputs["forecast"],  # type: ignore[arg-type]
        v7_paths=v7_paths,
        config=config,
    )

    assert tuple(paths) == tuple(CausalAlphaV8Candidate)
    assert (
        paths[CausalAlphaV8Candidate.V7_CONTROL].v6_target_path.digest
        == v7_paths[next(iter(v7_paths))].v6_target_path.digest
    )
    np.testing.assert_allclose(
        paths[
            CausalAlphaV8Candidate.ROBUST_CALIBRATED
        ].v6_target_path.expected_returns_4h,
        v7_paths[list(v7_paths)[2]].v6_target_path.expected_returns_4h,
    )
