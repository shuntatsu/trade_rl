"""Regression checks for bounded RSI scratch on fragmented native histories."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events


def _events(
    close: np.ndarray, present: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return calculate_feature_events(
        FeatureSpec(name="rsi", kind=FeatureKind.RSI, lookback=period),
        open_price=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones_like(close),
        funding_rate=np.zeros_like(close),
        funding_available=np.zeros(len(close), dtype=np.bool_),
        row_present=present,
        active=np.ones(len(close), dtype=np.bool_),
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("period", [1, 14, 35])
def test_rsi_gap_segments_equal_isolated_histories(dtype: type, period: int) -> None:
    rng = np.random.default_rng(1357)
    close = (1000 + rng.integers(-100, 101, 150)).astype(dtype)
    segments = [(0, 1), (3, 18), (20, 58), (62, 100), (104, 150)]
    present = np.zeros(len(close), dtype=np.bool_)
    for start, stop in segments:
        present[start:stop] = True
    expected = (
        np.zeros(len(close)),
        np.zeros(len(close), dtype=np.bool_),
        np.full(len(close), -1, dtype=np.int64),
    )
    for start, stop in segments:
        values, valid, source = _events(
            close[start:stop], np.ones(stop - start, dtype=np.bool_), period
        )
        expected[0][start:stop] = values
        expected[1][start:stop] = valid
        expected[2][start:stop] = np.where(valid, source + start, -1)
    actual = _events(close, present, period)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_rsi_gap_scratch_scales_with_bars_not_segment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Count ufunc input elements, rather than asserting machine-dependent timing.
    # Each gain/loss scratch operation should touch only its own contiguous segment.
    size = 4096
    close = 100.0 + (np.arange(size) % 31) / 10.0
    present = np.arange(size) % 32 != 31
    original_maximum = np.maximum
    visited: list[int] = []

    def maximum(left: np.ndarray, right: float) -> np.ndarray:
        visited.append(np.size(left))
        return original_maximum(left, right)

    monkeypatch.setattr(np, "maximum", maximum)
    values, valid, source = _events(close, present, 14)
    assert np.isfinite(values).all()
    assert np.count_nonzero(valid) == (size // 32) * 17
    assert np.all(source[~valid] == -1)
    assert visited
    assert sum(visited) <= 2 * size
