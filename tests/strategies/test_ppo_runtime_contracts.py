from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.strategies.test_ppo_feature_normalization import _fit
from tests.strategies.test_ppo_intent import SequenceStrategy, market
from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
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
    assert list(tmp_path.glob(".policy.staging-*")) == []


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


def test_economic_termination_keeps_finite_reward_and_terminal_observation() -> None:
    base = market()
    prices = np.asarray([[100.0], [100.0], [300.0], [300.0]])
    dataset = replace(
        base,
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
    )
    execution_cost = replace(
        ExecutionCostConfig.zero(),
        maintenance_margin_rate=0.25,
    )
    intents = (
        PositionIntent.SHORT,
        PositionIntent.SHORT,
        PositionIntent.SHORT,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=execution_cost,
    )
    env.reset(seed=7)

    observed_returns: list[float] = []
    observed_rewards: list[float] = []
    terminated = False
    for action in (0, 0, 0):
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert truncated is False
        observed_rewards.append(reward)
        observed_returns.append(float(info["interval_net_return"]))
        if terminated:
            break

    assert terminated is True
    assert env.book.termination_reason is not None
    assert env.book.termination_reason == replay.book.termination_reason
    np.testing.assert_allclose(observed_returns, replay.returns.values)
    np.testing.assert_allclose(observed_rewards, np.log1p(replay.returns.values))
    np.testing.assert_allclose(env.book.quantities, replay.book.quantities)

    with pytest.raises(RuntimeError, match="terminated"):
        env.step(1)


def test_explicit_reseed_restarts_sequential_symbol_schedule() -> None:
    env = PPOTradingEnv(
        pooled_market(),
        feature_indices=(0,),
        symbol_indices=(0, 1),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    first_observation, first_info = env.reset(seed=41)
    _, second_info = env.reset()
    restarted_observation, restarted_info = env.reset(seed=41)

    assert first_info["symbol_index"] == restarted_info["symbol_index"] == 0
    assert second_info["symbol_index"] == 1
    np.testing.assert_array_equal(first_observation, restarted_observation)


def test_directional_ppo_reward_charges_dataset_borrow_like_canonical_replay() -> None:
    base = market()
    dataset = replace(
        base,
        open=np.full_like(base.open, 100.0),
        high=np.full_like(base.high, 100.0),
        low=np.full_like(base.low, 100.0),
        close=np.full_like(base.close, 100.0),
        borrow_rate=np.full_like(base.close, 0.365),
    )
    intents = (
        PositionIntent.SHORT,
        PositionIntent.SHORT,
        PositionIntent.SHORT,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=3)

    observed_rewards = [env.step(0)[1] for _ in range(3)]

    assert replay.diagnostics.borrow_cost > 0.0
    assert sum(observed_rewards) < 0.0
    np.testing.assert_allclose(observed_rewards, np.log1p(replay.returns.values))


def test_ppo_reward_charges_dataset_funding_like_canonical_replay() -> None:
    base = market()
    dataset = replace(
        base,
        open=np.full_like(base.open, 100.0),
        high=np.full_like(base.high, 100.0),
        low=np.full_like(base.low, 100.0),
        close=np.full_like(base.close, 100.0),
        funding_rate=np.full_like(base.close, 0.001),
    )
    intents = (
        PositionIntent.LONG,
        PositionIntent.LONG,
        PositionIntent.LONG,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=5)

    observed_rewards = [env.step(2)[1] for _ in range(3)]

    assert replay.diagnostics.funding_pnl != 0.0
    np.testing.assert_allclose(observed_rewards, np.log1p(replay.returns.values))
    assert env.book.funding_pnl == pytest.approx(replay.book.funding_pnl)
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)


def test_ppo_reward_matches_dividend_and_cash_interest_carry() -> None:
    base = market()
    dataset = replace(
        base,
        open=np.full_like(base.open, 100.0),
        high=np.full_like(base.high, 100.0),
        low=np.full_like(base.low, 100.0),
        close=np.full_like(base.close, 100.0),
        dividend=np.full_like(base.close, 0.25),
        cash_rate=np.full(base.n_bars, 0.05, dtype=np.float64),
    )
    intents = (
        PositionIntent.LONG,
        PositionIntent.LONG,
        PositionIntent.LONG,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=7)

    observed_rewards = [env.step(2)[1] for _ in range(3)]

    assert sum(replay.returns.values) > 0.0
    np.testing.assert_allclose(observed_rewards, np.log1p(replay.returns.values))
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)


def test_directional_zero_overlay_keeps_dataset_fee_and_spread_costs() -> None:
    base = market()
    dataset = replace(
        base,
        open=np.full_like(base.open, 100.0),
        high=np.full_like(base.high, 100.0),
        low=np.full_like(base.low, 100.0),
        close=np.full_like(base.close, 100.0),
        fee_rate=np.full_like(base.close, 0.001),
        spread_rate=np.full_like(base.close, 0.002),
    )
    intents = (
        PositionIntent.LONG,
        PositionIntent.LONG,
        PositionIntent.FLAT,
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(intents),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=11)

    observed_rewards = [env.step(action)[1] for action in (2, 2, 1)]

    assert replay.diagnostics.total_cost > 0.0
    assert sum(observed_rewards) < 0.0
    np.testing.assert_allclose(observed_rewards, np.log1p(replay.returns.values))
    assert env.book.total_cost == pytest.approx(replay.book.total_cost)


def test_ppo_step_info_exposes_realized_risk_execution_and_carry_state() -> None:
    base = market()
    dataset = replace(
        base,
        funding_rate=np.full_like(base.close, 0.001),
        borrow_rate=np.full_like(base.close, 0.1),
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(max_participation_rate=1e-6),
        risk_config=PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=0.2,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        ),
    )
    env.reset(seed=13)

    _, _, _, _, info = env.step(2)

    assert info["realized_weight"] == pytest.approx(env.book.weights[0])
    assert info["was_constrained"] is True
    assert "max_turnover" in info["risk_reasons"]
    assert float(info["interval_cost_amount"]) > 0.0
    assert float(info["interval_funding_amount"]) != 0.0
    assert float(info["requested_turnover"]) > float(info["filled_turnover"])
    assert 0.0 <= float(info["fill_ratio"]) < 1.0
    assert info["termination_reason"] is None


def test_directional_zero_overlay_keeps_dataset_participation_capacity() -> None:
    base = market()
    dataset = replace(
        base,
        max_participation_rate=np.full_like(base.close, 1e-6),
    )
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy(
            (PositionIntent.LONG, PositionIntent.LONG, PositionIntent.LONG)
        ),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.5,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=17)

    _, _, _, _, first_info = env.step(2)
    env.step(2)
    env.step(2)

    assert float(first_info["fill_ratio"]) < 1.0
    assert env.book.quantities[0] == pytest.approx(replay.book.quantities[0])
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)


@pytest.mark.parametrize(
    ("dataset_changes", "intent", "action"),
    [
        (
            {"borrow_available": np.zeros((4, 1), dtype=np.bool_)},
            PositionIntent.SHORT,
            0,
        ),
        (
            {"buy_allowed": np.zeros((4, 1), dtype=np.bool_)},
            PositionIntent.LONG,
            2,
        ),
        (
            {"minimum_notional": np.full((4, 1), 10_000.0)},
            PositionIntent.LONG,
            2,
        ),
    ],
)
def test_ppo_target_does_not_bypass_market_admission_constraints(
    dataset_changes: dict[str, np.ndarray],
    intent: PositionIntent,
    action: int,
) -> None:
    dataset = replace(market(), **dataset_changes)
    replay = run_single_symbol_replay(
        dataset,
        SequenceStrategy((intent, intent, intent)),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env = PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        start_index=0,
        stop_index=3,
        gross_budget=0.1,
        initial_capital=1_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
    )
    env.reset(seed=19)

    _, _, _, _, info = env.step(action)
    env.step(action)
    env.step(action)

    assert abs(float(info["target_weight"])) > 0.0
    assert float(info["realized_weight"]) == pytest.approx(0.0)
    assert env.book.quantities[0] == pytest.approx(0.0)
    assert replay.book.quantities[0] == pytest.approx(0.0)
    assert env.book.portfolio_value == pytest.approx(replay.book.portfolio_value)
