"""Synthetic-only regressions for the frozen causal trend-shape feature semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events

_SHAPE_KINDS = (FeatureKind.LINEAR_REGRESSION_SLOPE, FeatureKind.TREND_R2)


def _shape(
    kind: FeatureKind,
    close: np.ndarray,
    *,
    present: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = close.size
    return calculate_feature_events(
        FeatureSpec(name=kind.value, kind=kind, lookback=24),
        open_price=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones(size, dtype=np.float64),
        funding_rate=np.zeros(size, dtype=np.float64),
        funding_available=np.zeros(size, dtype=np.bool_),
        row_present=(
            np.ones(size, dtype=np.bool_) if present is None else present
        ),
        active=np.ones(size, dtype=np.bool_) if active is None else active,
    )


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
def test_24bar_shape_uses_24_closes_not_24_price_intervals(
    kind: FeatureKind,
) -> None:
    close = 100.0 * np.exp(0.003 * np.arange(26, dtype=np.float64))
    values, valid, source_start = _shape(kind, close)
    np.testing.assert_array_equal(np.flatnonzero(valid), [23, 24, 25])
    np.testing.assert_array_equal(source_start[23:], [0, 1, 2])
    np.testing.assert_array_equal(source_start[:23], -np.ones(23, dtype=np.int64))

    # A 24-interval log return at t=24 still needs close[0].
    # Shape at t=24 uses only close[1:25] (23 intervals).
    different_first_close = close.copy()
    different_first_close[0] *= 1.5
    after, after_valid, after_start = _shape(kind, different_first_close)
    assert not math.isclose(
        math.log(close[24] / close[0]),
        math.log(different_first_close[24] / different_first_close[0]),
    )
    np.testing.assert_array_equal(values[24:], after[24:])
    np.testing.assert_array_equal(valid, after_valid)
    np.testing.assert_array_equal(source_start, after_start)


def test_equal_24_interval_returns_can_have_distinct_shape() -> None:
    x = np.arange(25, dtype=np.float64)
    smooth_log = np.linspace(0.0, math.log(1.05), 25)
    choppy_log = smooth_log + 0.04 * np.sin(2.0 * math.pi * 3.0 * x / 24.0)
    smooth = 100.0 * np.exp(smooth_log)
    choppy = 100.0 * np.exp(choppy_log)

    assert math.log(smooth[-1] / smooth[0]) == pytest.approx(
        math.log(choppy[-1] / choppy[0]), abs=1e-14
    )
    smooth_slope, _, _ = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, smooth)
    choppy_slope, _, _ = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, choppy)
    smooth_r2, smooth_valid, _ = _shape(FeatureKind.TREND_R2, smooth)
    choppy_r2, choppy_valid, _ = _shape(FeatureKind.TREND_R2, choppy)

    assert smooth_valid[24] and choppy_valid[24]
    assert smooth_r2[24] == pytest.approx(1.0, abs=1e-10)
    assert choppy_r2[24] < 0.3
    assert smooth_slope[24] != pytest.approx(choppy_slope[24])


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
@pytest.mark.parametrize("missing_field", ("present", "active"))
def test_missing_row_invalidates_every_affected_shape_window(
    kind: FeatureKind, missing_field: str
) -> None:
    close = 100.0 * np.exp(0.002 * np.arange(60, dtype=np.float64))
    mask = np.ones(60, dtype=np.bool_)
    mask[17] = False
    kwargs = {missing_field: mask}
    values, valid, start = _shape(kind, close, **kwargs)
    np.testing.assert_array_equal(np.flatnonzero(valid), np.arange(41, 60))
    np.testing.assert_array_equal(start[:41], -np.ones(41, dtype=np.int64))
    assert start[41] == 18
    assert values[41] == pytest.approx(
        0.002 if kind is FeatureKind.LINEAR_REGRESSION_SLOPE else 1.0,
        abs=1e-10,
    )


@pytest.mark.parametrize("kind", _SHAPE_KINDS)
def test_poisoning_future_rows_cannot_change_past_shape(kind: FeatureKind) -> None:
    x = np.arange(60, dtype=np.float64)
    close = 100.0 * np.exp(0.002 * x + 0.02 * np.sin(x / 4.0))
    poisoned = close.copy()
    poisoned[36:] *= 7.0
    base = _shape(kind, close)
    changed = _shape(kind, poisoned)
    for previous, following in zip(base, changed, strict=True):
        np.testing.assert_array_equal(previous[:36], following[:36])


def test_flat_and_signed_linear_trends_preserve_documented_convention() -> None:
    x = np.arange(30, dtype=np.float64)
    flat = np.full(30, 100.0)
    positive = 100.0 * np.exp(0.004 * x)
    negative = 100.0 * np.exp(-0.004 * x)

    for kind in _SHAPE_KINDS:
        flat_values, valid, start = _shape(kind, flat)
        assert valid[23:].all()
        assert start[23] == 0
        np.testing.assert_array_equal(flat_values[23:], np.zeros(7))

    up_slope = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, positive)[0][-1]
    down_slope = _shape(FeatureKind.LINEAR_REGRESSION_SLOPE, negative)[0][-1]
    assert up_slope == pytest.approx(0.004, abs=1e-12)
    assert down_slope == pytest.approx(-0.004, abs=1e-12)
    assert _shape(FeatureKind.TREND_R2, positive)[0][-1] == pytest.approx(1.0)
    assert _shape(FeatureKind.TREND_R2, negative)[0][-1] == pytest.approx(1.0)
