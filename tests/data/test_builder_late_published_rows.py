"""A late-published base row is hidden until ``available_at``, then usable.

Regression for an audit witness: the base builder treated a row whose
``available_at`` is after its timestamp as a permanent gap, so every feature
window containing it stayed invalid even after publication. The native path
already aligns events by publication time; the base path now does the same.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_N = 30
_LATE = 10
_DELAY_HOURS = 2


def _raw(*, delayed: bool, close_scale: float = 1.0) -> RawMarketSeries:
    rng = np.random.default_rng(11)
    timestamps = np.datetime64(_T0.replace(tzinfo=None), "ns") + np.arange(
        _N
    ) * np.timedelta64(1, "h")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, _N)))
    close[_LATE] *= close_scale
    open_price = np.concatenate((close[:1], close[:-1]))
    available_at = timestamps.copy()
    if delayed:
        available_at[_LATE] += np.timedelta64(_DELAY_HOURS, "h")
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=open_price,
        high=np.maximum(open_price, close) * 1.002,
        low=np.minimum(open_price, close) * 0.998,
        close=close,
        volume=rng.lognormal(5.0, 0.3, _N),
        funding_rate=np.zeros(_N),
        tradable=np.ones(_N, dtype=np.bool_),
    )


_FEATURES = (
    FeatureSpec("ret", FeatureKind.LOG_RETURN),
    FeatureSpec("vol5", FeatureKind.REALIZED_VOLATILITY, lookback=5),
    FeatureSpec("ema", FeatureKind.EMA_SLOPE, lookback=4),
)


def _build(raw: RawMarketSeries) -> MarketDataset:
    config = MarketBuildConfig(base_timeframe="1h", features=_FEATURES)
    contract = InstrumentContract("BTCUSDT", listed_at=_T0)
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}), (contract,)
    )


def test_late_row_becomes_usable_from_its_publication_time() -> None:
    on_time = _build(_raw(delayed=False))
    late = _build(_raw(delayed=True))
    published = _LATE + _DELAY_HOURS
    assert late.feature_available is not None
    assert late.feature_staleness_hours is not None
    assert on_time.feature_staleness_hours is not None

    # Before publication the row and every event depending on it are hidden:
    # the latest event wholly published before the late row is carried.
    for row in range(_LATE, published):
        np.testing.assert_array_equal(late.features[row], on_time.features[_LATE - 1])
        np.testing.assert_allclose(
            late.feature_staleness_hours[row, 0], float(row - (_LATE - 1))
        )

    # From publication onward the late row is part of every later window, so
    # the features equal the on-time build (no permanent hole).
    np.testing.assert_array_equal(
        late.features[published:], on_time.features[published:]
    )
    np.testing.assert_array_equal(
        late.feature_available[published:], on_time.feature_available[published:]
    )
    np.testing.assert_array_equal(
        late.feature_staleness_hours[published:],
        on_time.feature_staleness_hours[published:],
    )
    assert late.feature_available[_LATE + 4, 0, 1]
    assert late.feature_staleness_hours[_LATE + 4, 0, 1] == 0.0


@pytest.mark.parametrize("scale", (0.5, 1.7))
def test_late_row_value_cannot_reach_decisions_before_publication(
    scale: float,
) -> None:
    base = _build(_raw(delayed=True))
    mutated = _build(_raw(delayed=True, close_scale=scale))
    published = _LATE + _DELAY_HOURS

    assert base.feature_available is not None and mutated.feature_available is not None
    np.testing.assert_array_equal(
        base.features[:published], mutated.features[:published]
    )
    np.testing.assert_array_equal(
        base.feature_available[:published], mutated.feature_available[:published]
    )
    assert not np.array_equal(base.features[published], mutated.features[published])


def test_late_row_still_hidden_from_its_own_decision_row() -> None:
    late = _build(_raw(delayed=True))
    assert late.information_available is not None
    assert not late.information_available[_LATE, 0]
    assert late.information_available[_LATE + 1, 0]
