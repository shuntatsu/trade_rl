from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from tests.serving.test_shared_observation_builder import market_dataset
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.market_inputs import MarketInputResolver
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
    resolver = MarketInputResolver(trend_strategy=trend)
    env = ResidualMarketEnv(
        market,
        market_input_resolver=resolver,
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
    return market, builder, resolver, value


def _activate_matching(runtime: ServingRuntime, root: Path) -> ObservationInput:
    market, builder, resolver, value = _state()
    assert runtime.market_input_resolver.digest == resolver.digest
    runtime.activate(
        create_bundle(
            root,
            dataset_id=market.dataset_id,
            observation_schema_digest=builder.schema_digest(market),
            observation_size=builder.layout(market).size,
            market_inputs_digest=resolver.digest,
        )
    )
    return value


def test_predict_state_accepts_matching_dataset_and_schema(tmp_path: Path) -> None:
    _, _, resolver, _ = _state()
    runtime = ServingRuntime(market_input_resolver=resolver)
    value = _activate_matching(runtime, tmp_path / "matching")

    np.testing.assert_array_equal(
        runtime.predict_state(value),
        np.zeros(2, dtype=np.float32),
    )


def test_predict_state_rejects_dataset_identity_mismatch(tmp_path: Path) -> None:
    market, builder, resolver, value = _state()
    runtime = ServingRuntime(
        observation_builder=builder,
        market_input_resolver=resolver,
    )
    runtime.activate(
        create_bundle(
            tmp_path / "wrong-dataset",
            dataset_id="f" * 64,
            observation_schema_digest=builder.schema_digest(market),
            observation_size=builder.layout(market).size,
            market_inputs_digest=resolver.digest,
        )
    )

    with pytest.raises(ValueError, match="dataset identity"):
        runtime.predict_state(value)


def test_predict_state_rejects_observation_schema_mismatch(tmp_path: Path) -> None:
    market, builder, resolver, value = _state()
    runtime = ServingRuntime(
        observation_builder=builder,
        market_input_resolver=resolver,
    )
    runtime.activate(
        create_bundle(
            tmp_path / "wrong-schema",
            dataset_id=market.dataset_id,
            observation_schema_digest="f" * 64,
            observation_size=builder.layout(market).size,
            market_inputs_digest=resolver.digest,
        )
    )

    with pytest.raises(ValueError, match="observation schema"):
        runtime.predict_state(value)


def test_predict_rejects_wrong_observation_vector_size(tmp_path: Path) -> None:
    runtime = ServingRuntime()
    snapshot = runtime.activate(create_bundle(tmp_path / "size", observation_size=5))

    with pytest.raises(ValueError, match="observation size"):
        runtime.predict(
            np.zeros(4, dtype=np.float32),
            dataset_id=snapshot.dataset_id,
            observation_schema_digest=snapshot.observation_schema_digest,
            market_inputs_digest=snapshot.market_inputs_digest,
        )


def test_activation_rejects_market_input_resolver_mismatch(tmp_path: Path) -> None:
    runtime = ServingRuntime()

    with pytest.raises(ValueError, match="market inputs"):
        runtime.activate(
            create_bundle(
                tmp_path / "wrong-market-inputs",
                market_inputs_digest="f" * 64,
            )
        )


def test_predict_state_recomputes_untrusted_trend_and_alpha(tmp_path: Path) -> None:
    market, builder, resolver, value = _state()
    runtime = ServingRuntime(
        observation_builder=builder,
        market_input_resolver=resolver,
    )
    runtime.activate(
        create_bundle(
            tmp_path / "recompute",
            dataset_id=market.dataset_id,
            observation_schema_digest=builder.schema_digest(market),
            observation_size=builder.layout(market).size,
            market_inputs_digest=resolver.digest,
        )
    )
    poisoned = ObservationInput(
        dataset=value.dataset,
        index=value.index,
        trends=type(value.trends)(
            fast=np.ones(market.n_symbols),
            base=-np.ones(market.n_symbols),
            slow=np.ones(market.n_symbols),
        ),
        alpha=np.ones(market.n_symbols),
        hybrid=value.hybrid,
        shadow=value.shadow,
        start_index=value.start_index,
        end_index=value.end_index,
        hybrid_risk_scale=value.hybrid_risk_scale,
        shadow_risk_scale=value.shadow_risk_scale,
    )

    np.testing.assert_array_equal(
        runtime.build_observation(poisoned),
        runtime.build_observation(value),
    )
