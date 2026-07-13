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
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _market(*, next_tradable: bool = True, suspended_symbol: int | None = None):
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        40
    ) * np.timedelta64(1, "h")
    values: dict[str, RawMarketSeries] = {}
    symbols = ("UP", "MID", "DOWN")
    slopes = (0.003, 0.001, -0.002)
    for symbol_index, (symbol, slope) in enumerate(zip(symbols, slopes)):
        close = np.exp(np.arange(40, dtype=np.float64) * slope)
        open_price = np.concatenate([close[:1], close[:-1]])
        tradable = np.ones(40, dtype=np.bool_)
        tradable[17] = next_tradable
        if suspended_symbol == symbol_index:
            tradable[14] = False
        values[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=open_price,
            high=np.maximum(open_price, close) * 1.001,
            low=np.minimum(open_price, close) * 0.999,
            close=close,
            volume=np.full(40, 1_000.0),
            funding_rate=np.zeros(40),
            tradable=tradable,
        )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    contracts = tuple(
        InstrumentContract(
            symbol=symbol,
            listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for symbol in symbols
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource(values), contracts
    )


def _environment(market, *, alpha_provider=None, alpha_enabled: bool = False):
    return ResidualMarketEnv(
        market,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        alpha_provider=alpha_provider,
        alpha_enabled=alpha_enabled,
        config=ResidualMarketEnvConfig(
            episode_bars=12,
            decision_every=4,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_current_pretrade_target_does_not_depend_on_next_bar_tradability() -> None:
    proposal = np.array([0.3, -0.2, -0.1], dtype=np.float64)
    resolved: list[np.ndarray] = []
    for next_tradable in (True, False):
        env = _environment(_market(next_tradable=next_tradable))
        env.reset(options={"start_idx": 16})
        resolved.append(env._constrain_target(proposal, env.hybrid).weights)

    np.testing.assert_allclose(resolved[0], resolved[1], atol=0.0, rtol=0.0)


def test_trend_excludes_symbol_with_suspension_inside_lookback() -> None:
    market = _market(suspended_symbol=1)
    trend = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    )

    targets = trend.targets(market, 16)

    assert targets.base[1] == 0.0
    assert targets.slow[1] == 0.0
    assert targets.base[0] != 0.0
    assert targets.base[2] != 0.0


def test_alpha_is_zeroed_outside_current_eligible_universe() -> None:
    market = _market()
    tradable = market.tradable.copy()
    tradable[16, 1] = False
    object.__setattr__(market, "tradable", tradable)

    env = _environment(
        market,
        alpha_provider=lambda dataset, index: np.array([0.2, 0.6, 0.2]),
        alpha_enabled=True,
    )
    env.reset(options={"start_idx": 16})

    np.testing.assert_allclose(env._alpha_at(16), np.array([0.2, 0.0, 0.2]))
