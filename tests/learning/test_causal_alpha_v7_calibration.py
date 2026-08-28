from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_teacher import CausalAlphaRidgeModel
from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
)
from trade_rl.learning.causal_alpha_v7_calibration import (
    CausalAlphaV7CalibrationFit,
    CausalAlphaV7CalibrationRows,
    fit_causal_alpha_v7_calibration,
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


def _rows(symbol: str, *, count: int = 160) -> CausalAlphaV7CalibrationRows:
    decisions = np.arange(500, 500 + count, dtype=np.int64)
    phase = 0.3 if symbol == "BTCUSDT" else 0.9
    x = decisions.astype(np.float64)
    fast = 0.01 * np.sin(x * 0.071 + phase)
    direction = np.sin(x * 0.053 + phase)
    features = np.column_stack(
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
    realized = 0.006 * np.sin(x * 0.083 + phase) + 0.25 * fast
    return CausalAlphaV7CalibrationRows(
        symbol=symbol,
        decision_indices=decisions,
        label_end_indices=decisions + 1,
        features=features,
        feature_available=np.ones(features.shape, dtype=np.bool_),
        realized_returns=realized,
        range_digest=_range().digest,
    )


def _fit(
    rows: dict[str, CausalAlphaV7CalibrationRows] | None = None,
) -> CausalAlphaV7CalibrationFit:
    return fit_causal_alpha_v7_calibration(
        rows={
            "BTCUSDT": _rows("BTCUSDT"),
            "SOLUSDT": _rows("SOLUSDT"),
        }
        if rows is None
        else rows,
        calibration_range=_range(),
        config=CausalAlphaV7CalibrationConfig(),
    )


def test_v7_calibration_is_deterministic_across_symbol_mapping_order() -> None:
    first_rows = {
        "BTCUSDT": _rows("BTCUSDT"),
        "SOLUSDT": _rows("SOLUSDT"),
    }
    second_rows = dict(reversed(tuple(first_rows.items())))

    first = _fit(first_rows)
    second = _fit(second_rows)

    assert first.digest == second.digest
    assert first.per_symbol_support == (("BTCUSDT", 160), ("SOLUSDT", 160))
    assert first.positive_direction_support > 0
    assert first.negative_direction_support > 0


def test_v7_calibration_rejects_future_or_wrong_range_rows() -> None:
    valid = _rows("BTCUSDT")
    ends = valid.label_end_indices.copy()
    ends[-1] = _range().train_stop
    with pytest.raises(ValueError, match="strictly before train stop"):
        _fit(
            {
                "BTCUSDT": replace(valid, label_end_indices=ends, digest=""),
                "SOLUSDT": _rows("SOLUSDT"),
            }
        )

    with pytest.raises(ValueError, match="range digest"):
        _fit(
            {
                "BTCUSDT": replace(valid, range_digest=_digest("f"), digest=""),
                "SOLUSDT": _rows("SOLUSDT"),
            }
        )


def test_v7_calibration_requires_both_realized_directions() -> None:
    rows = {
        symbol: replace(
            value,
            realized_returns=np.abs(value.realized_returns) + 0.001,
            digest="",
        )
        for symbol, value in {
            "BTCUSDT": _rows("BTCUSDT"),
            "SOLUSDT": _rows("SOLUSDT"),
        }.items()
    }

    with pytest.raises(ValueError, match="both realized directions"):
        _fit(rows)


def test_v7_calibration_prediction_is_block_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit()
    original = CausalAlphaRidgeModel.predict
    observed: list[int] = []

    def recording_predict(
        self: CausalAlphaRidgeModel,
        features: object,
        *,
        feature_available: object | None = None,
    ) -> np.ndarray:
        observed.append(len(np.asarray(features)))
        return original(self, features, feature_available=feature_available)

    monkeypatch.setattr(CausalAlphaRidgeModel, "predict", recording_predict)
    base = _rows("BTCUSDT").features
    features = np.tile(base, (33, 1))[:5_000]
    available = np.ones(features.shape, dtype=np.bool_)

    calibrated_return, reliability = fit.predict(
        features,
        feature_available=available,
    )

    assert calibrated_return.shape == (5_000,)
    assert reliability.shape == (5_000,)
    assert observed
    assert max(observed) <= 4_096
