from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_u1_base_env,
    make_u1_feature_specs,
    make_u1_market,
    make_u1_wrapper,
)
from trade_rl.rl.environment_config import EpisodeBoundaryMode
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.rl.universal_single_instrument_env import EpisodeRoutedSingleInstrumentEnv
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder
from trade_rl.rl.universal_trade_reward import universal_net_log_growth_reward
from trade_rl.simulation.execution import ExecutionCostConfig


def _contract() -> UniversalTradePolicyContract:
    return UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())


def _wrapper(*, base=None) -> UniversalTradeEnvironment:
    if base is None:
        return make_u1_wrapper()
    return UniversalTradeEnvironment(base, contract=_contract())


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


def test_u1_wrapper_accepts_fixed_contract_and_exposes_only_policy_surface() -> None:
    env = _wrapper()

    observation, _info = env.reset(seed=7, options={"initial_state_mode": "cash"})

    assert env.observation_space.contains(observation)
    assert env.action_space.contains(np.asarray([0.25], dtype=np.float32))
    assert "policy_state" in observation
    assert "instrument_context" not in observation
    assert "symbol" not in observation


@pytest.mark.parametrize(
    "config_change",
    (
        {"accept_legacy_actions": True},
        {"signal_delay_decisions": 0},
        {"decision_hours": 0.5},
        {"initial_state_modes": ("baseline",)},
        {
            "episode_boundary_mode": EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION,
            "finite_horizon_observation": True,
        },
    ),
)
def test_u1_wrapper_rejects_environment_contract_drift(
    config_change: dict[str, object],
) -> None:
    base = make_u1_base_env()
    object.__setattr__(base, "config", replace(base.config, **config_change))

    with pytest.raises(ValueError, match="U1|contract|config"):
        _wrapper(base=base)


def test_u1_wrapper_rejects_non_pure_growth_reward() -> None:
    base = make_u1_base_env()
    reward_config = replace(
        base.config.resolved_reward_config(),
        projection_penalty_weight=0.1,
        terminal_equity_weight=1.0,
    )
    object.__setattr__(
        base, "config", replace(base.config, reward_config=reward_config)
    )

    with pytest.raises(ValueError, match="reward|U1|contract"):
        _wrapper(base=base)


@pytest.mark.parametrize(
    "horizon_override",
    (
        {"episode_hours": 24.0},
        {"episode_bars": 96},
    ),
)
def test_u1_wrapper_rejects_reset_horizon_override(
    horizon_override: dict[str, object],
) -> None:
    env = _wrapper()

    with pytest.raises(ValueError, match="U1|horizon|episode|contract"):
        env.reset(
            seed=17,
            options={
                "initial_state_mode": "cash",
                "start_idx": 6000,
                **horizon_override,
            },
        )


def test_u1_wrapper_is_cash_only_and_action_strict() -> None:
    env = _wrapper()

    with pytest.raises(ValueError, match="cash|U1"):
        env.reset(options={"initial_state_mode": "baseline"})

    env.reset(seed=11, options={"initial_state_mode": "cash"})
    with pytest.raises(ValueError):
        env.step(np.asarray([1.0001], dtype=np.float32))


def test_u1_wrapper_preserves_submission_pending_risk_and_realized_state() -> None:
    contract = _contract()
    env = make_u1_wrapper(
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
    base = env.base_env
    env.reset(
        seed=19,
        options={"start_idx": 6000, "initial_state_mode": "cash"},
    )
    env.step(np.asarray([0.60], dtype=np.float32))
    observation, _reward, _terminated, _truncated, _info = env.step(
        np.asarray([0.80], dtype=np.float32)
    )

    state = _policy_state(observation, contract=contract)
    assert state["policy_requested_weight"] == pytest.approx(0.80)
    assert state["pending_target_active"] == pytest.approx(1.0)
    assert state["pending_target_weight"] == pytest.approx(0.80)
    assert state["risk_projected_weight"] == pytest.approx(0.35)
    assert 0.0 < abs(state["current_weight"]) < abs(state["risk_projected_weight"])
    assert 0.0 <= state["fill_ratio"] < 1.0


def test_u1_wrapper_external_truncation_does_not_liquidate_open_position() -> None:
    env = _wrapper()
    base = env.base_env
    env.reset(
        seed=23,
        options={"start_idx": 6000, "initial_state_mode": "cash"},
    )
    env.step(np.asarray([0.70], dtype=np.float32))
    env.step(np.asarray([0.70], dtype=np.float32))
    assert abs(float(base.hybrid.weights[0])) > 1e-12

    object.__setattr__(base, "end_index", base.current_index + 1)
    _observation, _reward, terminated, truncated, info = env.step(
        np.asarray([0.70], dtype=np.float32)
    )

    assert terminated is False
    assert truncated is True
    assert abs(float(base.hybrid.weights[0])) > 1e-12
    assert "hybrid_liquidation" not in info
    assert info["terminal_accounting_mode"] == "mark_to_market"
    assert info["terminal_liquidation_cost"] == pytest.approx(0.0)


def test_u1_wrapper_recomputes_wealth_reward_and_rejects_base_reward_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_u1_base_env()
    env = _wrapper(base=base)
    env.reset(seed=13, options={"initial_state_mode": "cash"})

    before = float(base.hybrid.portfolio_value)
    _observation, reward, _terminated, _truncated, _info = env.step(
        np.asarray([0.6], dtype=np.float32)
    )
    after = float(base.hybrid.portfolio_value)
    expected = universal_net_log_growth_reward(before_value=before, after_value=after)
    assert reward == pytest.approx(expected, abs=1e-10, rel=0.0)

    original_step = base.step
    calls = 0

    def drifting_step(action: np.ndarray):
        nonlocal calls
        calls += 1
        observation, delegated_reward, terminated, truncated, info = original_step(
            action
        )
        return observation, delegated_reward + 1e-4, terminated, truncated, info

    monkeypatch.setattr(base, "step", drifting_step)
    with pytest.raises(RuntimeError, match="reward|drift|U1"):
        env.step(np.asarray([0.8], dtype=np.float32))
    assert calls == 1


def test_u1_wrapper_is_compatible_with_two_symbol_episode_router() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    datasets = {symbol: make_u1_market(symbol=symbol) for symbol in symbols}
    bindings = tuple(
        InstrumentDatasetBinding(
            concrete_symbol=symbol,
            source_dataset_id=datasets[symbol].dataset_id,
            symbol_dataset_digest=datasets[symbol].dataset_id,
            execution_metadata_digest=f"{index + 1:x}" * 64,
            instrument_descriptor_digest=f"{index + 3:x}" * 64,
            split="train",
        )
        for index, symbol in enumerate(symbols)
    )
    contract = _contract()
    environments: dict[str, UniversalTradeEnvironment] = {}

    def environment_factory(
        binding: InstrumentDatasetBinding,
    ) -> UniversalTradeEnvironment:
        environment = make_u1_wrapper(
            dataset=datasets[binding.concrete_symbol],
            contract=contract,
        )
        environments[binding.concrete_symbol] = environment
        return environment

    routed = EpisodeRoutedSingleInstrumentEnv(
        train_symbols=symbols,
        partition_digest="a" * 64,
        bindings=bindings,
        environment_factory=environment_factory,
        run_seed=29,
        environment_index=0,
    )

    first_observation, _first_info = routed.reset(
        seed=29,
        options={"start_idx": 6000, "initial_state_mode": "cash"},
    )
    first_symbol = routed.active_episode_binding.dataset_binding.concrete_symbol
    assert routed.observation_space.contains(first_observation)
    assert "instrument_context" not in first_observation

    first_environment = environments[first_symbol]
    object.__setattr__(
        first_environment.base_env,
        "end_index",
        first_environment.current_index + 1,
    )
    _observation, _reward, terminated, truncated, _info = routed.step(
        np.asarray([0.25], dtype=np.float32)
    )
    assert terminated is False
    assert truncated is True

    second_observation, _second_info = routed.reset(
        seed=29,
        options={"start_idx": 6000, "initial_state_mode": "cash"},
    )
    second_symbol = routed.active_episode_binding.dataset_binding.concrete_symbol
    assert second_symbol != first_symbol
    assert routed.observation_space.contains(second_observation)
    assert "instrument_context" not in second_observation
    assert set(environments) == set(symbols)

    second_environment = environments[second_symbol]
    object.__setattr__(
        second_environment.base_env,
        "end_index",
        second_environment.current_index + 1,
    )
    _observation, _reward, terminated, truncated, _info = routed.step(
        np.asarray([-0.25], dtype=np.float32)
    )
    assert terminated is False
    assert truncated is True
