"""Literal warm-up of the legacy ADX and directional-index features.

Within one contiguous usable segment of length n starting at s, Wilder's
recursion seeds TR and +DM/-DM averages at offset L - 1 (L rows), so +DI,
-DI and their spread are first defined at s + L - 1. DX exists from that
offset; the ADX seed is the mean of the first L DX values, i.e. offsets
L - 1 .. 2L - 2, so the first ADX row is s + 2L - 2 and needs exactly
2L - 1 rows. A segment's availability may depend only on its own rows up to
t, never on whether later rows exist. The oracle below is a plain-Python
transcription of that recursion, independent of the production helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events

_DIRECTIONAL = (FeatureKind.PLUS_DI, FeatureKind.MINUS_DI, FeatureKind.DI_SPREAD)


def _prices(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.004, n)))
    return {"open": open_, "high": high, "low": low, "close": close}


def _calculate(
    kind: FeatureKind, lookback: int, p: dict[str, np.ndarray], present: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(present)
    return calculate_feature_events(
        FeatureSpec(kind.value, kind, lookback=lookback),
        open_price=p["open"],
        high=p["high"],
        low=p["low"],
        close=p["close"],
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        row_present=present,
        active=np.ones(n, dtype=bool),
    )


def _segment_oracle(
    high: list[float], low: list[float], close: list[float], period: int
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Return {offset: (+DI, -DI)} and {offset: ADX} for one segment."""
    n = len(close)
    tr = [high[0] - low[0]]
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, n):
        tr.append(
            max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        )
        up, down = high[i] - high[i - 1], low[i - 1] - low[i]
        plus_dm.append(up if up > down and up > 0.0 else 0.0)
        minus_dm.append(down if down > up and down > 0.0 else 0.0)

    def wilder(values: list[float]) -> dict[int, float]:
        result: dict[int, float] = {}
        if n < period:
            return result
        average = math.fsum(values[:period]) / period
        result[period - 1] = average
        for i in range(period, n):
            average = ((period - 1) * average + values[i]) / period
            result[i] = average
        return result

    atr, plus_avg, minus_avg = wilder(tr), wilder(plus_dm), wilder(minus_dm)
    di = {
        i: (plus_avg[i] / atr[i], minus_avg[i] / atr[i]) for i in atr if atr[i] > 1e-12
    }
    dx = {}
    for i, (plus, minus) in di.items():
        total = 100.0 * (plus + minus)
        dx[i] = 0.0 if total <= 1e-12 else 100.0 * 100.0 * abs(plus - minus) / total
    adx: dict[int, float] = {}
    first = 2 * period - 2
    if n > first and all(i in dx for i in range(period - 1, first + 1)):
        value = math.fsum(dx[i] for i in range(period - 1, first + 1)) / period
        adx[first] = value / 100.0
        for i in range(first + 1, n):
            if i not in dx:
                break
            value = ((period - 1) * value + dx[i]) / period
            adx[i] = value / 100.0
    return di, adx


def _layout(lookback: int, tail: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Segments of 2L-2, 2L-1 and 2L usable rows, each closed by a gap."""
    lengths = (2 * lookback - 2, 2 * lookback - 1, 2 * lookback, tail)
    present: list[bool] = []
    segments = []
    for length in lengths:
        segments.append((len(present), len(present) + length))
        present.extend([True] * length)
        present.append(False)
    return np.array(present[:-1], dtype=bool), segments


@pytest.mark.parametrize("lookback", [2, 3, 5, 14])
@pytest.mark.parametrize("kind", [FeatureKind.ADX, FeatureKind.ADX_CAUSAL])
def test_adx_first_row_needs_exactly_two_lookback_minus_one_rows(kind, lookback):
    present, segments = _layout(lookback, tail=3 * lookback)
    p = _prices(len(present), seed=lookback)
    values, valid, source_start = _calculate(kind, lookback, p, present)

    expected_valid = np.zeros(len(present), dtype=bool)
    for start, stop in segments:
        _, adx = _segment_oracle(
            list(p["high"][start:stop]),
            list(p["low"][start:stop]),
            list(p["close"][start:stop]),
            lookback,
        )
        for offset, value in adx.items():
            expected_valid[start + offset] = True
            assert values[start + offset] == pytest.approx(value, rel=1e-12, abs=0)
            assert source_start[start + offset] == start
    np.testing.assert_array_equal(valid, expected_valid)

    short, exact, full, _ = segments
    first = 2 * lookback - 2
    assert not valid[short[0] : short[1]].any()  # 2L-2 rows: nothing
    assert np.flatnonzero(valid[exact[0] : exact[1]]).tolist() == [first]  # 2L-1
    assert np.flatnonzero(valid[full[0] : full[1]]).tolist() == [first, first + 1]
    assert np.all(source_start[~valid] == -1)


@pytest.mark.parametrize("lookback", [2, 3, 5, 14])
@pytest.mark.parametrize("kind", _DIRECTIONAL)
def test_directional_indices_first_row_needs_exactly_lookback_rows(kind, lookback):
    present, segments = _layout(lookback, tail=lookback + 2)
    p = _prices(len(present), seed=100 + lookback)
    values, valid, source_start = _calculate(kind, lookback, p, present)

    expected_valid = np.zeros(len(present), dtype=bool)
    for start, stop in segments:
        di, _ = _segment_oracle(
            list(p["high"][start:stop]),
            list(p["low"][start:stop]),
            list(p["close"][start:stop]),
            lookback,
        )
        assert min(di) == lookback - 1
        for offset, (plus, minus) in di.items():
            plus, minus = min(max(plus, 0.0), 1.0), min(max(minus, 0.0), 1.0)
            expected = {
                FeatureKind.PLUS_DI: plus,
                FeatureKind.MINUS_DI: minus,
                FeatureKind.DI_SPREAD: plus - minus,
            }[kind]
            expected_valid[start + offset] = True
            assert values[start + offset] == pytest.approx(expected, rel=1e-12, abs=0)
            assert source_start[start + offset] == start
    np.testing.assert_array_equal(valid, expected_valid)
    assert np.all(source_start[~valid] == -1)


@pytest.mark.parametrize(
    "kind", [FeatureKind.ADX, FeatureKind.ADX_CAUSAL, *_DIRECTIONAL]
)
@pytest.mark.parametrize("lookback", [3, 14])
def test_directional_availability_never_depends_on_later_rows(kind, lookback):
    present, _ = _layout(lookback, tail=3 * lookback)
    p = _prices(len(present), seed=200 + lookback)
    full = _calculate(kind, lookback, p, present)
    for cut in range(1, len(present) + 1):
        prefix = _calculate(
            kind, lookback, {k: v[:cut] for k, v in p.items()}, present[:cut]
        )
        for left, right in zip(full, prefix, strict=True):
            np.testing.assert_array_equal(left[:cut], right, err_msg=f"cut={cut}")


def test_flat_seed_window_matches_oracle_for_adx_and_directional_indices():
    lookback = 3
    n = 30
    p = _prices(n, seed=5)
    for key in p:
        p[key][:4] = 100.0  # zero true range: the first Wilder ATRs are zero
    present = np.ones(n, dtype=bool)
    di, adx = _segment_oracle(
        list(p["high"]), list(p["low"]), list(p["close"]), lookback
    )
    # DX is undefined inside the ADX seed window, so this segment has no ADX.
    assert not adx and min(di) > lookback - 1
    _, adx_valid, _ = _calculate(FeatureKind.ADX, lookback, p, present)
    assert not adx_valid.any()
    for kind in _DIRECTIONAL:
        _, valid, _ = _calculate(kind, lookback, p, present)
        np.testing.assert_array_equal(np.flatnonzero(valid), sorted(di))
