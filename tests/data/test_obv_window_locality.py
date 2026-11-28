"""OBV kinds depend only on rows inside their declared ``[source_start, t]``."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events

_OBV_KINDS = (
    FeatureKind.OBV_SLOPE,
    FeatureKind.OBV_CHANGE,
    FeatureKind.OBV_ACCELERATION,
)


def _series(n: int, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    for index in rng.choice(np.arange(1, n), size=n // 10, replace=False):
        close[index] = close[index - 1]  # zero-direction OBV rows
    open_price = close * (1.0 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_price, close) * 1.003
    low = np.minimum(open_price, close) * 0.997
    volume = rng.lognormal(7.0, 1.0, n)
    return {
        "open_price": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _run(
    kind: FeatureKind,
    lookback: int,
    data: dict[str, np.ndarray],
    row_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(data["close"])
    return calculate_feature_events(
        FeatureSpec(kind.value, kind, lookback=lookback),
        open_price=data["open_price"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=np.bool_),
        row_present=row_present,
        active=np.ones(n, dtype=np.bool_),
    )


@pytest.mark.parametrize("kind", _OBV_KINDS)
@pytest.mark.parametrize("lookback", (1, 3, 9, 24))
def test_obv_value_is_bit_identical_when_history_starts_at_source_start(
    kind: FeatureKind, lookback: int
) -> None:
    n = 240
    data = _series(n)
    row_present = np.ones(n, dtype=np.bool_)
    row_present[[57, 140]] = False
    values, valid, source_start = _run(kind, lookback, data, row_present)
    assert valid.any()
    for index in np.flatnonzero(valid):
        start = int(source_start[index])
        assert 0 <= start <= index
        window = slice(start, index + 1)
        truncated = {key: value[window] for key, value in data.items()}
        t_values, t_valid, t_source = _run(
            kind, lookback, truncated, row_present[window]
        )
        assert t_valid[-1], (kind, lookback, index)
        assert int(t_source[-1]) == 0
        assert t_values[-1].tobytes() == values[index].tobytes(), (
            kind,
            lookback,
            index,
            float(values[index]),
            float(t_values[-1]),
        )


def _literal(kind: FeatureKind, lookback: int, close, volume):
    data = {
        "open_price": np.asarray(close, dtype=np.float64),
        "high": np.asarray(close, dtype=np.float64),
        "low": np.asarray(close, dtype=np.float64),
        "close": np.asarray(close, dtype=np.float64),
        "volume": np.asarray(volume, dtype=np.float64),
    }
    return _run(kind, lookback, data, np.ones(len(close), dtype=np.bool_))


@pytest.mark.parametrize("kind", (FeatureKind.OBV_SLOPE, FeatureKind.OBV_CHANGE))
def test_obv_slope_and_change_literal_oracle(kind: FeatureKind) -> None:
    # signed flows for rows 1..4: +2, -3, 0, +5
    values, valid, source_start = _literal(
        kind, 2, [10.0, 11.0, 10.0, 10.0, 12.0], [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert valid.tolist() == [False, False, True, True, True]
    assert source_start.tolist()[2:] == [0, 1, 2]
    assert values[2] == float(Fraction(-1, 5))
    assert values[3] == float(Fraction(-3, 7))
    assert values[4] == float(Fraction(5, 9))


def test_obv_acceleration_literal_oracle() -> None:
    values, valid, source_start = _literal(
        FeatureKind.OBV_ACCELERATION,
        2,
        [10.0, 11.0, 10.0, 10.0, 12.0, 11.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    assert valid.tolist() == [False] * 4 + [True, True]
    assert source_start.tolist()[4:] == [0, 1]
    # t=4: recent (+0 +5) - previous (+2 -3) = 6 over volume 2+3+4+5
    assert values[4] == float(Fraction(6, 14))
    # t=5: recent (+5 -6) - previous (-3 +0) = 2 over volume 3+4+5+6
    assert values[5] == float(Fraction(2, 18))


@pytest.mark.parametrize("kind", _OBV_KINDS)
def test_obv_is_unaffected_by_large_flow_before_source_start(
    kind: FeatureKind,
) -> None:
    # A 1e17 flow at row 1 lies outside the window of the last row.  A running
    # segment total would absorb the later unit flows (ulp(1e17) == 16).
    close = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    volume = [0.0, 1e17, 1.0, 1.0, 1.0, 1.0, 3.0]
    values, valid, source_start = _literal(kind, 2, close, volume)
    assert valid[-1]
    assert source_start[-1] >= 2
    if kind is FeatureKind.OBV_ACCELERATION:
        # recent (+1 +3) - previous (+1 +1) over volume 1+1+1+3
        assert values[-1] == float(Fraction(2, 6))
    else:
        assert values[-1] == 1.0
