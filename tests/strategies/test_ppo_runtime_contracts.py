from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.strategies.test_ppo_feature_normalization import _fit
from tests.strategies.test_ppo_intent import SequenceStrategy, market
from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, PPOTradingEnv
from trade_rl.strategies.rl.ppo_artifact import save_normalized_ppo


class _ActionPolicy:
    def __init__(self, action: object) -> None:
        self.action = action

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]:
        del observation
        assert deterministic is True
        return self.action, None


class _FailingSavePolicy(_ActionPolicy):
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"partial policy")
        raise RuntimeError("simulated policy serialization failure")


def _observation() -> StrategyObservation:
    return StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01", "ns"),
        symbol="BTCUSDT",
        features=np.asarray([0.5]),
        feature_available=np.asarray([True]),
        feature_staleness=np.asarray([0.0], dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float64),
        global_feature_available=np.ones(1, dtype=np.bool_),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )


@pytest.mark.parametrize(
    "action",
    [
        1.0,
        np.float64(1.0),
        "1",
        np.asarray([1.0]),
    ],
)
def test_policy_rejects_non_integer_action_aliases(action: object) -> None:
    strategy = PPOIntentStrategy(_ActionPolicy(action), feature_indices=(0,))

    with pytest.raises(ValueError, match="integer"):
        strategy.decide(_observation())


def _stochastic_episode_rewards(seed: int) -> tuple[float, float, float]:
    env = PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(slippage_std=0.01),
    )
    env.reset(seed=seed)
    first = env.step(2)[1]
    env.reset()
    second = env.step(2)[1]
    env.reset()
    third = env.step(2)[1]
    return first, second, third


def test_sequential_stochastic_execution_seed_advances_across_episode_resets() -> None:
    first = _stochastic_episode_rewards(17)
    second = _stochastic_episode_rewards(17)

    np.testing.assert_array_equal(first, second)
    assert first[1] != first[2]


def test_explicit_reseed_restarts_stochastic_episode_stream() -> None:
    env = PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(slippage_std=0.01),
    )

    env.reset(seed=23)
    first = env.step(2)[1]
    env.reset()
    env.step(2)
    env.reset(seed=23)
    restarted = env.step(2)[1]

    assert restarted == first


def test_failed_normalized_policy_save_leaves_no_partial_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "policy"
    strategy = PPOIntentStrategy(
        _FailingSavePolicy(1),
        feature_indices=(0,),
        feature_normalizer=_fit(),
    )

    with pytest.raises(RuntimeError, match="serialization"):
        save_normalized_ppo(root, strategy)

    assert not root.exists()


def test_env_matches_canonical_replay_with_cost_capacity_and_turnover_risk() -> None:
    dataset = market()
    intents = (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.FLAT)
    execution_cost = ExecutionCostConfig(max_participation_rate=1e-6)
    risk_config = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=0.2,
        drawdown_start=1.0,
        drawdown_stop=1.0,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk=PreTradeRisk(risk_config),
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
        risk_config=risk_config,
    )
    env.reset(seed=7)

    observed_returns: list[float] = []
    observed_rewards: list[float] = []
    for action in (2, 2, 1):
        _, reward, terminated, truncated, info = env.step(action)
        observed_rewards.append(reward)
        observed_returns.append(float(info["interval_net_return"]))
        assert truncated is False
    assert terminated is True

    np.testing.assert_allclose(observed_returns, replay.returns.values)
    np.testing.assert_allclose(
        observed_rewards,
        np.log1p(replay.returns.values),
    )
    np.testing.assert_allclose(env.book.quantities, replay.book.quantities)
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
    assert replay.diagnostics.total_cost > 0.0
    assert replay.book.fill_count > 0
    assert replay.book.quantities[0] > 0.0
