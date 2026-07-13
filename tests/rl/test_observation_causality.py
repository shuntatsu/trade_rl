from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.rl.observations import ObservationBuilder, ObservationInput
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def raw_series(n_bars: int, *, next_tradable: bool = True, future_shift: float = 0.0) -> RawMarketSeries:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    close[21:] *= np.exp(future_shift)
    open_price = np.concatenate([close[:1], close[:-1]])
    tradable = np.ones(n_bars, dtype=np.bool_)
    tradable[21] = next_tradable
    return RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full(n_bars, 100.0),
        funding_rate=np.zeros(n_bars),
        tradable=tradable,
    )


def dataset(*, next_tradable: bool = True, future_shift: float = 0.0):
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="ret_1", kind=FeatureKind.LOG_RETURN, lookback=1),
            FeatureSpec(
                name="funding_bps",
                kind=FeatureKind.FUNDING_BPS,
                max_staleness_hours=8.0,
            ),
        ),
    )
    source = InMemoryMarketDataSource(
        {
            "A": raw_series(
                64, next_tradable=next_tradable, future_shift=future_shift
            ),
            "B": raw_series(
                64, next_tradable=next_tradable, future_shift=-future_shift
            ),
        }
    )
    contracts = (
        InstrumentContract(
            symbol="A", listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
        InstrumentContract(
            symbol="B", listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        ),
    )
    return MarketDatasetBuilder(config).build(source, contracts)


def observation(market) -> np.ndarray:
    index = 20
    trends = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    ).targets(market, index)
    hybrid = BookState.zero(market.n_symbols, 1.0, market.close[index])
    shadow = BookState.zero(market.n_symbols, 1.0, market.close[index])
    return ObservationBuilder().build(
        ObservationInput(
            dataset=market,
            index=index,
            trends=trends,
            alpha=np.zeros(market.n_symbols),
            hybrid=hybrid,
            shadow=shadow,
            start_index=8,
            end_index=40,
            hybrid_risk_scale=1.0,
            shadow_risk_scale=1.0,
        )
    )


def test_next_bar_tradability_is_not_visible_at_decision_time() -> None:
    np.testing.assert_array_equal(
        observation(dataset(next_tradable=True)),
        observation(dataset(next_tradable=False)),
    )


def test_future_market_mutation_does_not_change_current_observation() -> None:
    np.testing.assert_array_equal(
        observation(dataset(future_shift=0.0)),
        observation(dataset(future_shift=0.5)),
    )


def test_observation_contains_per_feature_masks_and_staleness() -> None:
    market = dataset()
    vector = observation(market)
    layout = ObservationBuilder().layout(market)
    per_symbol = vector[: market.n_symbols * layout.per_symbol_width].reshape(
        market.n_symbols, layout.per_symbol_width
    )
    n = market.n_features

    np.testing.assert_array_equal(per_symbol[:, n : 2 * n], market.feature_available[20])
    np.testing.assert_allclose(per_symbol[:, 2 * n : 3 * n], market.feature_staleness[20])
    np.testing.assert_array_equal(per_symbol[:, 3 * n], market.tradable[20])
    np.testing.assert_array_equal(per_symbol[:, 3 * n + 1], market.symbol_active[20])
