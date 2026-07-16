from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events

_NEW_KINDS = (
    FeatureKind.BODY_RETURN,
    FeatureKind.HIGH_LOW_RANGE,
    FeatureKind.UPPER_WICK_RATIO,
    FeatureKind.LOWER_WICK_RATIO,
    FeatureKind.CLOSE_LOCATION_VALUE,
    FeatureKind.GAP_RETURN,
    FeatureKind.VOLUME_LOG_CHANGE,
    FeatureKind.PARKINSON_VOLATILITY,
    FeatureKind.GARMAN_KLASS_VOLATILITY,
    FeatureKind.DOWNSIDE_VOLATILITY,
    FeatureKind.UPSIDE_VOLATILITY,
    FeatureKind.VOLATILITY_OF_VOLATILITY,
    FeatureKind.RANGE_EXPANSION,
    FeatureKind.ATR_CHANGE,
    FeatureKind.PLUS_DI,
    FeatureKind.MINUS_DI,
    FeatureKind.DI_SPREAD,
    FeatureKind.EMA_DISTANCE,
    FeatureKind.EMA_SLOPE,
    FeatureKind.LINEAR_REGRESSION_SLOPE,
    FeatureKind.TREND_R2,
    FeatureKind.MFI,
    FeatureKind.CMF,
    FeatureKind.VWAP_DISTANCE,
    FeatureKind.PRICE_VOLUME_CORRELATION,
    FeatureKind.OBV_CHANGE,
    FeatureKind.OBV_ACCELERATION,
    FeatureKind.RELATIVE_VOLUME,
    FeatureKind.FUNDING_CHANGE,
    FeatureKind.FUNDING_ZSCORE,
)


def _inputs(*, mutate_future: bool = False) -> dict[str, np.ndarray]:
    n = 128
    phase = np.arange(n, dtype=np.float64)
    close = 100.0 * np.exp(0.0015 * phase + 0.015 * np.sin(phase / 7.0))
    open_price = close * (1.0 + 0.002 * np.sin(phase / 5.0))
    high = np.maximum(open_price, close) * (1.0 + 0.004 + 0.001 * np.sin(phase / 3.0))
    low = np.minimum(open_price, close) * (1.0 - 0.004 - 0.001 * np.cos(phase / 4.0))
    volume = 1_000.0 * (1.0 + 0.3 * np.sin(phase / 6.0)) + 10.0 * phase
    funding_available = phase.astype(int) % 8 == 0
    funding_rate = np.where(funding_available, 0.0001 * np.sin(phase / 13.0), 0.0)
    if mutate_future:
        close[81:] *= 5.0
        open_price[81:] *= 0.2
        high[81:] *= 8.0
        low[81:] *= 0.1
        volume[81:] *= 100.0
        funding_rate[81:] += 0.05
        funding_available[81:] = True
    return {
        "open_price": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "funding_rate": funding_rate,
        "funding_available": funding_available,
        "row_present": np.ones(n, dtype=np.bool_),
        "active": np.ones(n, dtype=np.bool_),
    }


def _calculate(kind: FeatureKind, *, mutate_future: bool = False, lookback: int = 20):
    return calculate_feature_events(
        FeatureSpec(
            name=kind.value,
            kind=kind,
            lookback=lookback,
            min_periods=min(lookback, 4),
        ),
        **_inputs(mutate_future=mutate_future),
    )


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_new_features_are_prefix_causal(kind: FeatureKind) -> None:
    baseline = _calculate(kind)
    mutated = _calculate(kind, mutate_future=True)
    for left, right in zip(baseline, mutated, strict=True):
        np.testing.assert_array_equal(left[:81], right[:81])


def test_candle_geometry_uses_only_current_and_previous_completed_bar() -> None:
    inputs = _inputs()
    index = 50
    body, body_valid, body_start = _calculate(FeatureKind.BODY_RETURN, lookback=1)
    range_, range_valid, _ = _calculate(FeatureKind.HIGH_LOW_RANGE, lookback=1)
    clv, clv_valid, _ = _calculate(FeatureKind.CLOSE_LOCATION_VALUE, lookback=1)
    gap, gap_valid, gap_start = _calculate(FeatureKind.GAP_RETURN, lookback=1)

    expected_body = np.log(inputs["close"][index] / inputs["open_price"][index])
    expected_range = (inputs["high"][index] - inputs["low"][index]) / inputs["close"][
        index
    ]
    expected_clv = (
        2.0 * inputs["close"][index] - inputs["high"][index] - inputs["low"][index]
    ) / (inputs["high"][index] - inputs["low"][index])
    expected_gap = np.log(inputs["open_price"][index] / inputs["close"][index - 1])

    assert (
        body_valid[index]
        and range_valid[index]
        and clv_valid[index]
        and gap_valid[index]
    )
    assert body_start[index] == index
    assert gap_start[index] == index - 1
    assert body[index] == pytest.approx(expected_body)
    assert range_[index] == pytest.approx(expected_range)
    assert clv[index] == pytest.approx(expected_clv)
    assert gap[index] == pytest.approx(expected_gap)


def test_directional_features_preserve_direction_information() -> None:
    plus, valid, _ = _calculate(FeatureKind.PLUS_DI, lookback=14)
    minus, minus_valid, _ = _calculate(FeatureKind.MINUS_DI, lookback=14)
    spread, spread_valid, _ = _calculate(FeatureKind.DI_SPREAD, lookback=14)
    index = int(np.flatnonzero(valid & minus_valid & spread_valid)[-1])
    assert -1.0 <= plus[index] <= 1.0
    assert -1.0 <= minus[index] <= 1.0
    assert spread[index] == pytest.approx(plus[index] - minus[index])


def test_volatility_and_flow_features_are_finite_and_scaled() -> None:
    for kind in (
        FeatureKind.PARKINSON_VOLATILITY,
        FeatureKind.GARMAN_KLASS_VOLATILITY,
        FeatureKind.DOWNSIDE_VOLATILITY,
        FeatureKind.UPSIDE_VOLATILITY,
        FeatureKind.MFI,
        FeatureKind.CMF,
        FeatureKind.PRICE_VOLUME_CORRELATION,
    ):
        values, valid, source_start = _calculate(kind, lookback=20)
        assert valid.any(), kind
        assert np.isfinite(values[valid]).all(), kind
        assert np.all(source_start[valid] >= 0), kind
        if kind in {
            FeatureKind.MFI,
            FeatureKind.CMF,
            FeatureKind.PRICE_VOLUME_CORRELATION,
        }:
            assert np.max(np.abs(values[valid])) <= 1.0 + 1e-12
