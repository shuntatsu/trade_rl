from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from tests.serving.test_shared_observation_builder import market_dataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationBuilder, ObservationInput
from trade_rl.serving.runtime import ServingRuntime
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _state():
    market = market_dataset()
    builder = ObservationBuilder()
    trend = TrendStrategy(
        TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
    )
    env = ResidualMarketEnv(
        market,
        trend_strategy=trend,
        config=ResidualMarketEnvConfig(
            episode_bars=24,
            decision_every=4,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(options={"start_idx": 16})
    trends, alpha = env._market_inputs()
    value = ObservationInput(
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
    return market, builder, value


def _activate_matching(runtime: ServingRuntime, root: Path) -> ObservationInput:
    market, builder, value = _state()
    runtime.activate(
        create_bundle(
            root,
            dataset_id=market.dataset_id,
            observation_schema_digest=builder.schema_digest(market),
            observation_size=builder.layout(market).size,
        )
    )
    return value


def test_predict_state_accepts_matching_dataset_and_schema(tmp_path: Path) -> None:
    runtime = ServingRuntime()
    value = _activate_matching(runtime, tmp_path / "matching")

    np.testing.assert_array_equal(
        runtime.predict_state(value),
        np.zeros(2, dtype=np.float32),
    )


def test_predict_state_rejects_dataset_identity_mismatch(tmp_path: Path) -> None:
    market, builder, value = _state()
    runtime = ServingRuntime(observation_builder=builder)
    runtime.activate(
        create_bundle(
            tmp_path / "wrong-dataset",
            dataset_id="f" * 64,
            observation_schema_digest=builder.schema_digest(market),
            observation_size=builder.layout(market).size,
        )
    )

    with pytest.raises(ValueError, match="dataset identity"):
        runtime.predict_state(value)


def test_predict_state_rejects_observation_schema_mismatch(tmp_path: Path) -> None:
    market, builder, value = _state()
    runtime = ServingRuntime(observation_builder=builder)
    runtime.activate(
        create_bundle(
            tmp_path / "wrong-schema",
            dataset_id=market.dataset_id,
            observation_schema_digest="f" * 64,
            observation_size=builder.layout(market).size,
        )
    )

    with pytest.raises(ValueError, match="observation schema"):
        runtime.predict_state(value)


def test_predict_rejects_wrong_observation_vector_size(tmp_path: Path) -> None:
    runtime = ServingRuntime()
    runtime.activate(create_bundle(tmp_path / "size", observation_size=5))

    with pytest.raises(ValueError, match="observation size"):
        runtime.predict(np.zeros(4, dtype=np.float32))
