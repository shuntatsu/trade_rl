from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_u1_feature_specs,
    make_u1_market,
    make_u1_wrapper,
)
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder
from trade_rl.simulation.execution import ExecutionCostConfig


def _policy_state(
    observation: dict[str, np.ndarray],
    *,
    contract: UniversalTradePolicyContract,
) -> dict[str, float]:
    fields = UniversalTradeObservationBuilder(contract=contract).policy_state_fields
    return {
        field: float(value)
        for field, value in zip(fields, observation["policy_state"], strict=True)
    }


def _assert_reward_telescopes(
    *,
    rewards: list[float],
    initial_value: float,
    final_value: float,
) -> None:
    assert sum(rewards) / 100.0 == pytest.approx(
        math.log(final_value / initial_value),
        abs=1e-10,
        rel=0.0,
    )


def test_flat_market_cost_reward_equals_realized_wealth_only() -> None:
    execution = ExecutionCostConfig(
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0002,
        max_participation_rate=1.0,
        maintenance_margin_rate=0.0,
    )
    wrapper = make_u1_wrapper(
        dataset=make_u1_market(price_drift=0.0, volume=1_000_000_000.0),
        execution_cost=execution,
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = float(wrapper.base_env.hybrid.portfolio_value)
    rewards: list[float] = []

    for action in (0.8, 0.0, 0.0):
        _observation, reward, _terminated, _truncated, _info = wrapper.step(
            np.asarray([action], dtype=np.float32)
        )
        rewards.append(reward)

    final = float(wrapper.base_env.hybrid.portfolio_value)
    assert wrapper.base_env.hybrid.total_cost > 0.0
    assert final < initial
    _assert_reward_telescopes(
        rewards=rewards,
        initial_value=initial,
        final_value=final,
    )


def test_funding_is_accounted_once_in_wealth_reward() -> None:
    wrapper = make_u1_wrapper(
        dataset=make_u1_market(
            price_drift=0.0,
            volume=1_000_000_000.0,
            funding_rate_value=0.001,
            funding_due_from=6002,
        ),
        execution_cost=ExecutionCostConfig.zero(),
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = float(wrapper.base_env.hybrid.portfolio_value)
    rewards: list[float] = []

    for _ in range(4):
        _observation, reward, _terminated, _truncated, _info = wrapper.step(
            np.asarray([0.8], dtype=np.float32)
        )
        rewards.append(reward)

    final = float(wrapper.base_env.hybrid.portfolio_value)
    assert wrapper.base_env.hybrid.funding_pnl < 0.0
    assert final < initial
    _assert_reward_telescopes(
        rewards=rewards,
        initial_value=initial,
        final_value=final,
    )


def test_short_borrow_is_accounted_once_in_wealth_reward() -> None:
    execution = replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0)
    wrapper = make_u1_wrapper(
        dataset=make_u1_market(
            price_drift=0.0,
            volume=1_000_000_000.0,
            borrow_rate_value=0.365,
            borrow_available_value=True,
        ),
        execution_cost=execution,
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = float(wrapper.base_env.hybrid.portfolio_value)
    rewards: list[float] = []

    for _ in range(4):
        _observation, reward, _terminated, _truncated, _info = wrapper.step(
            np.asarray([-0.8], dtype=np.float32)
        )
        rewards.append(reward)

    final = float(wrapper.base_env.hybrid.portfolio_value)
    assert wrapper.base_env.hybrid.borrow_cost > 0.0
    assert final < initial
    _assert_reward_telescopes(
        rewards=rewards,
        initial_value=initial,
        final_value=final,
    )


def test_cash_reset_clears_previous_episode_policy_and_execution_state() -> None:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    wrapper = make_u1_wrapper(contract=contract)
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    wrapper.step(np.asarray([0.7], dtype=np.float32))
    wrapper.step(np.asarray([-0.4], dtype=np.float32))

    observation, _info = wrapper.reset(
        options={"start_idx": 6100, "initial_state_mode": "cash"}
    )
    state = _policy_state(observation, contract=contract)

    assert state["current_weight"] == pytest.approx(0.0)
    assert state["policy_requested_weight"] == pytest.approx(0.0)
    assert state["previous_action"] == pytest.approx(0.0)
    assert state["pending_target_active"] == pytest.approx(0.0)
    assert state["pending_target_weight"] == pytest.approx(0.0)
    assert state["risk_projected_weight"] == pytest.approx(0.0)
    assert state["pending_notional_ratio"] == pytest.approx(0.0)
    assert state["position_age_days"] == pytest.approx(0.0)
    assert state["fill_ratio"] == pytest.approx(0.0)
    assert state["unfilled_turnover_ratio"] == pytest.approx(0.0)


def test_request_pending_risk_realized_and_order_pending_remain_distinct() -> None:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    wrapper = make_u1_wrapper(
        dataset=make_u1_market(volume=100.0),
        max_abs_weight=0.35,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.01,
            maintenance_margin_rate=0.0,
        ),
        contract=contract,
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    wrapper.step(np.asarray([0.60], dtype=np.float32))
    observation, _reward, _terminated, _truncated, _info = wrapper.step(
        np.asarray([0.80], dtype=np.float32)
    )
    state = _policy_state(observation, contract=contract)

    assert state["policy_requested_weight"] == pytest.approx(0.80)
    assert state["pending_target_active"] == pytest.approx(1.0)
    assert state["pending_target_weight"] == pytest.approx(0.80)
    assert state["risk_projected_weight"] == pytest.approx(0.35)
    assert 0.0 < abs(state["current_weight"]) < abs(state["risk_projected_weight"])
    assert 0.0 <= state["fill_ratio"] < 1.0
    assert state["pending_notional_ratio"] > 0.0

    next_observation, _reward, _terminated, _truncated, _info = wrapper.step(
        np.asarray([0.0], dtype=np.float32)
    )
    next_state = _policy_state(next_observation, contract=contract)

    assert next_state["policy_requested_weight"] == pytest.approx(0.0)
    assert next_state["pending_target_active"] == pytest.approx(1.0)
    assert next_state["pending_target_weight"] == pytest.approx(0.0)
    assert next_state["risk_projected_weight"] == pytest.approx(0.35)
    assert next_state["pending_notional_ratio"] > 0.0
