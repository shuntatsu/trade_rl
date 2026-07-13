from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.rl.observations import ObservationBuilder, ObservationInput
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

_INDEX = 20


def _raw(*, scale: float, mutate_future: bool) -> RawMarketSeries:
    n_bars = 64
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = scale * np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    open_price = np.concatenate([close[:1], close[:-1]])
    volume = 100.0 + np.arange(n_bars, dtype=np.float64)
    funding = np.zeros(n_bars, dtype=np.float64)
    tradable = np.ones(n_bars, dtype=np.bool_)
    available_at = timestamps.copy()
    if mutate_future:
        volume[_INDEX + 1 :] *= 1_000.0
        funding[_INDEX + 1 :] = 0.05
        tradable[_INDEX + 2 :: 2] = False
        available_at[_INDEX + 1 :] += np.timedelta64(12, "h")
    return RawMarketSeries(
        timestamps=timestamps,
        available_at=available_at,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=volume,
        funding_rate=funding,
        tradable=tradable,
    )


def _dataset(*, mutate_future: bool) -> MarketDataset:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),
            FeatureSpec(
                name="volume_z",
                kind=FeatureKind.VOLUME_ZSCORE,
                lookback=8,
                min_periods=4,
            ),
            FeatureSpec(
                name="funding_bps",
                kind=FeatureKind.FUNDING_BPS,
                max_staleness_hours=8.0,
            ),
        ),
    )
    source = InMemoryMarketDataSource(
        {
            "A": _raw(scale=1.0, mutate_future=mutate_future),
            "B": _raw(scale=2.0, mutate_future=mutate_future),
        }
    )
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for symbol in ("A", "B")
    )
    return MarketDatasetBuilder(config).build(source, contracts)


def _observation(dataset: MarketDataset) -> np.ndarray:
    trends = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    ).targets(dataset, _INDEX)
    hybrid = BookState.zero(dataset.n_symbols, 1.0, dataset.close[_INDEX])
    shadow = BookState.zero(dataset.n_symbols, 1.0, dataset.close[_INDEX])
    return ObservationBuilder().build(
        ObservationInput(
            dataset=dataset,
            index=_INDEX,
            trends=trends,
            alpha=np.zeros(dataset.n_symbols),
            hybrid=hybrid,
            shadow=shadow,
            start_index=8,
            end_index=40,
            hybrid_risk_scale=1.0,
            shadow_risk_scale=1.0,
        )
    )


def test_future_volume_funding_availability_and_tradability_are_not_observed() -> None:
    np.testing.assert_array_equal(
        _observation(_dataset(mutate_future=False)),
        _observation(_dataset(mutate_future=True)),
    )
