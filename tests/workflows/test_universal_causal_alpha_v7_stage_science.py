from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4Forecast,
    build_causal_alpha_v4_forecast,
)
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
)
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationRows,
    fit_causal_alpha_v7_calibration,
)
from trade_rl.workflows.universal_causal_alpha_v7_stage_science import (
    build_causal_alpha_v7_attribution_boundaries,
    build_causal_alpha_v7_calibration_rows,
    build_causal_alpha_v7_feature_matrix,
)


def _digest(char: str) -> str:
    return char * 64


def _forecast(rows: int = 40) -> CausalAlphaV4Forecast:
    x = np.arange(rows, dtype=np.float64)
    fast = 0.01 * np.sin(0.2 * x)
    return build_causal_alpha_v4_forecast(
        symbol="BTCUSDT",
        decision_indices=np.arange(500, 500 + rows, dtype=np.int64),
        beta=np.ones(rows),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions={"4h": fast * 0.4, "24h": fast, "72h": fast * 2.0},
        residual_predictions={"4h": fast * 0.6, "24h": fast, "72h": fast},
        direction_scores={
            "4h": np.sin(0.3 * x),
            "24h": np.cos(0.1 * x),
            "72h": np.sin(0.1 * x),
        },
        market_model_digests={h: _digest("a") for h in ("4h", "24h", "72h")},
        residual_model_digests={h: _digest("b") for h in ("4h", "24h", "72h")},
        direction_model_digests={h: _digest("c") for h in ("4h", "24h", "72h")},
        fit_digest=_digest("d"),
    )


def _state(rows: int = 40) -> SimpleNamespace:
    x = np.arange(rows, dtype=np.float64)
    return SimpleNamespace(
        realized_volatility=0.01 + x * 0.0001,
        liquidity=1.0 + x * 0.01,
        basis_positioning_stress=-0.2 + x * 0.01,
        state_eligible=np.ones(rows, dtype=np.bool_),
        actionable=np.ones(rows, dtype=np.bool_),
    )


def _range() -> CausalAlphaV7CalibrationRange:
    return CausalAlphaV7CalibrationRange(
        base_fit_cutoff=500,
        calibration_start=500,
        train_stop=1_000,
        block_boundaries=(500, 625, 750, 875, 1_000),
        split_digest=_digest("e"),
        feature_names=CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    )


def test_v7_feature_matrix_has_fixed_symbol_free_schema_and_masks_missing_state() -> None:
    forecast = _forecast()
    state = _state()
    state.realized_volatility[3] = np.nan
    state.state_eligible[3] = False
    uncertainty = {h: np.full(40, 0.01) for h in ("4h", "24h", "72h")}

    features, available = build_causal_alpha_v7_feature_matrix(
        forecast=forecast,
        uncertainty=uncertainty,
        state=state,
    )

    assert features.shape == (40, len(CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert not available[3].any()
    assert features[3, 3] == 0.0
    assert all("symbol" not in name for name in CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES)


def test_v7_calibration_rows_are_purged_on_4h_label_end() -> None:
    forecast = _forecast()
    state = _state()
    sample = SimpleNamespace(
        symbol="BTCUSDT",
        decision_indices=forecast.decision_indices,
        labels_4h=np.linspace(-0.02, 0.02, 40),
        label_end_indices_4h=np.arange(501, 541, dtype=np.int64),
    )
    sample.label_end_indices_4h[-3:] = 1_000
    uncertainty = {h: np.full(40, 0.01) for h in ("4h", "24h", "72h")}

    rows = build_causal_alpha_v7_calibration_rows(
        sample=sample,
        forecast=forecast,
        uncertainty=uncertainty,
        state=state,
        calibration_range=_range(),
    )

    assert rows.symbol == "BTCUSDT"
    assert len(rows.decision_indices) == 37
    assert np.all(rows.label_end_indices < _range().train_stop)


def test_v7_attribution_boundaries_use_only_calibration_rows() -> None:
    calibration_range = _range()
    config = CausalAlphaV7CalibrationConfig(minimum_pooled_support=256)
    records: dict[str, CausalAlphaV7CalibrationRows] = {}
    for symbol, phase in (("A", 0.0), ("B", 0.7)):
        count = 160
        x = np.arange(count, dtype=np.float64)
        features = np.column_stack(
            (
                0.01 * np.sin(x * 0.07 + phase),
                np.sin(x * 0.05 + phase),
                np.log(0.01 + x * 0.00001),
                0.01 + x * 0.0001,
                1.0 + x * 0.01,
                0.2 + x * 0.001,
                0.02 * np.sin(x * 0.03),
                0.03 * np.cos(x * 0.02),
                np.sin(x * 0.02),
                np.cos(x * 0.04),
            )
        )
        realized = 0.005 * np.sin(x * 0.11 + phase)
        records[symbol] = CausalAlphaV7CalibrationRows(
            symbol=symbol,
            decision_indices=np.arange(500, 500 + count),
            label_end_indices=np.arange(501, 501 + count),
            features=features,
            feature_available=np.ones_like(features, dtype=np.bool_),
            realized_returns=realized,
            range_digest=calibration_range.digest,
        )
    fit = fit_causal_alpha_v7_calibration(
        rows=records,
        calibration_range=calibration_range,
        config=config,
    )

    boundaries = build_causal_alpha_v7_attribution_boundaries(
        rows=records,
        fit=fit,
    )

    assert boundaries.calibration_range_digest == calibration_range.digest
    assert boundaries.confidence[0] < boundaries.confidence[1] < boundaries.confidence[2]
    assert boundaries.realized_volatility == tuple(
        np.quantile(np.concatenate([r.features[:, 3] for r in records.values()]), (0.25, 0.5, 0.75))
    )
