"""Structural scratch accounting and causal resets in fragmented histories."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.features import core

_NAMES = ("_atr", "_adx", "_directional_indices", "ema_distance", "ema_slope")


def _prices(n):
    close = 100.0 + np.arange(n) / 10 + np.sin(np.arange(n))
    return close + 2, close - 1, close


def _feature(name, high, low, close, usable, period):
    if name.startswith("ema_"):
        return core._ema_feature(close, usable, period, slope=name == "ema_slope")
    return getattr(core, name)(high, low, close, usable, period)


@pytest.mark.parametrize("name", _NAMES)
def test_segment_scratch_work_is_linear(name, monkeypatch):
    n = 4000
    high, low, close = _prices(n)
    usable = np.arange(n) % 9 != 8
    original_like, original_zeros = np.zeros_like, np.zeros
    allocations = []

    def counted_like(array, *args, **kwargs):
        result = original_like(array, *args, **kwargs)
        allocations.append(result.size)
        return result

    def counted_zeros(shape, *args, **kwargs):
        result = original_zeros(shape, *args, **kwargs)
        allocations.append(result.size)
        return result

    with monkeypatch.context() as m:
        m.setattr(core.np, "zeros_like", counted_like)
        m.setattr(core.np, "zeros", counted_zeros)
        results = _feature(name, high, low, close, usable, 3)
    assert results[-2].any()
    # Count allocated scratch elements, not host-dependent wall-clock time.
    assert sum(allocations) <= 12 * n


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("period", [1, 2, 3, 14, 50, 10**100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fragmented_features_equal_isolated_segments(name, period, dtype):
    high, low, close = (a.astype(dtype) for a in _prices(151))
    spans = [(0, 1), (2, 5), (6, 20), (23, 64), (65, 66), (68, 151)]
    usable = np.zeros(len(close), dtype=bool)
    for left, right in spans:
        usable[left:right] = True
    actual = _feature(name, high, low, close, usable, period)
    expected = [np.zeros_like(v) for v in actual]
    expected[-1].fill(-1)
    for left, right in spans:
        parts = _feature(
            name,
            high[left:right],
            low[left:right],
            close[left:right],
            np.ones(right - left, dtype=bool),
            period,
        )
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                part = np.where(part >= 0, part + left, -1)
            expected[index][left:right] = part
    for a, b in zip(actual, expected, strict=True):
        assert a.tobytes() == b.tobytes()


def test_literal_wilder_seed_and_gap():
    close = np.arange(100.0, 110.0)
    usable = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    high, low = close + 1, close - 1
    atr, atr_valid, atr_start = core._atr(high, low, close, usable, 2)
    plus, minus, di_valid, di_start = core._directional_indices(
        high, low, close, usable, 2
    )
    adx, adx_valid, adx_start = core._adx(high, low, close, usable, 2)
    expected_valid = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    expected_start = np.array([-1, 0, 0, 0, -1, -1, 5, 5, 5, 5])
    np.testing.assert_array_equal(atr_valid, expected_valid)
    np.testing.assert_array_equal(di_valid, expected_valid)
    np.testing.assert_array_equal(atr_start, expected_start)
    np.testing.assert_array_equal(di_start, expected_start)
    np.testing.assert_array_equal(atr[expected_valid], 2 / close[expected_valid])
    np.testing.assert_array_equal(
        plus, [0, 0.25, 0.375, 0.4375, 0, 0, 0.25, 0.375, 0.4375, 0.46875]
    )
    assert not minus.any()
    np.testing.assert_array_equal(adx_valid, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(adx_start, [-1, -1, 0, 0, -1, -1, -1, 5, 5, 5])
    np.testing.assert_array_equal(adx, adx_valid.astype(float))
