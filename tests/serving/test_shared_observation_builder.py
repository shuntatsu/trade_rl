from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract, MarketBuildConfig
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationBuilder, ObservationInput
from trade_rl.serving.runtime import ServingRuntime
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def market_dataset():
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        80
    ) * np.timedelta64(1, "h")
    values: dict[str, RawMarketSeries] = {}
    for symbol, slope in (("UP", 0.002), ("DOWN", -0.001)):
        close = np.exp(np.arange(80, dtype=np.float64) * slope)
        open_price = np.concatenate([close[:1], close[:-1]])
        values[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=open_price,
            high=np.maximum(open_price, close) * 1.001,
            low=np.minimum(open_price, close) * 0.999,
            close=close,
            volume=np.full(80, 1_000.0),
            funding_rate=np.zeros(80),
            tradable=np.ones(80, dtype=np.bool_),
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
        for symbol in ("UP", "DOWN")
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource(values), contracts
    )


def test_environment_and_serving_use_identical_observation_bytes() -> None:
    market = market_dataset()
    observation_builder = ObservationBuilder()
    trend = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    )
    env = ResidualMarketEnv(
        market,
        trend_strategy=trend,
        observation_builder=observation_builder,
        config=ResidualMarketEnvConfig(
            episode_bars=24,
            decision_every=4,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env_observation, _ = env.reset(options={"start_idx": 16})
    trends, alpha = env._market_inputs()
    structured = ObservationInput(
        dataset=market,
        index=env.current_index,
        trends=trends,
        alpha=alpha,
        hybrid=env.hybrid,
        shadow=env.shadow,
        start_index=env.start_index,
        end_index=env.end_index,
        hybrid_risk_scale=env.pre_trade_risk.risk_scale(env._drawdown(env.hybrid)),
        shadow_risk_scale=env.pre_trade_risk.risk_scale(env._drawdown(env.shadow)),
    )
    serving = ServingRuntime(observation_builder=observation_builder)

    serving_observation = serving.build_observation(structured)

    assert env_observation.dtype == np.float32
    assert serving_observation.tobytes() == env_observation.tobytes()
