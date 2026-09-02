from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import (
    make_u1_base_env,
    make_u1_feature_specs,
)
from trade_rl.rl.environment_config import EpisodeBoundaryMode
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.rl.universal_trade_reward import universal_net_log_growth_reward


def _wrapper(*, base=None) -> UniversalTradeEnvironment:
    resolved_base = make_u1_base_env() if base is None else base
    return UniversalTradeEnvironment(
        resolved_base,
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
    )


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
def test_u1_wrapper_rejects_environment_contract_drift(config_change: dict[str, object]) -> None:
    base = make_u1_base_env()
    object.__setattr__(base, "config", replace(base.config, **config_change))

    with pytest.raises(ValueError, match="U1|contract|config"):
        _wrapper(base=base)


def test_u1_wrapper_rejects_non_pure_growth_reward() -> None:
    base = make_u1_base_env()
    reward_config = replace(base.config.resolved_reward_config(), projection_penalty_weight=0.1)
    object.__setattr__(base, "config", replace(base.config, reward_config=reward_config))

    with pytest.raises(ValueError, match="reward|U1|contract"):
        _wrapper(base=base)


def test_u1_wrapper_is_cash_only_and_action_strict() -> None:
    env = _wrapper()

    with pytest.raises(ValueError, match="cash|U1"):
        env.reset(options={"initial_state_mode": "baseline"})

    env.reset(seed=11, options={"initial_state_mode": "cash"})
    with pytest.raises(ValueError):
        env.step(np.asarray([1.0001], dtype=np.float32))


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
        observation, delegated_reward, terminated, truncated, info = original_step(action)
        return observation, delegated_reward + 1e-4, terminated, truncated, info

    monkeypatch.setattr(base, "step", drifting_step)
    with pytest.raises(RuntimeError, match="reward|drift|U1"):
        env.step(np.asarray([0.8], dtype=np.float32))
    assert calls == 1
