from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_policy_schedule_contract import (
    EnvironmentPolicyScheduleContract,
    EnvironmentPolicyScheduleContractBuilder,
)
from trade_rl.rl.rewards import AbsoluteGrowthRewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _market() -> MarketDataset:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 120.0, n_bars),
            np.linspace(100.0, 90.0, n_bars),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _config(**overrides: object) -> ResidualMarketEnvConfig:
    values: dict[str, object] = {
        "initial_capital": 100_000.0,
        "episode_bars": 8,
        "decision_every": 2,
        "reward": AbsoluteGrowthRewardConfig(),
        "execution_cost": ExecutionCostConfig.zero(),
    }
    values.update(overrides)
    return ResidualMarketEnvConfig(**values)  # type: ignore[arg-type]


def _build(
    *,
    config: ResidualMarketEnvConfig | None,
    action_spec: ActionSpec | None = None,
    pre_trade_risk: PreTradeRisk | None = None,
    alpha_enabled: bool = False,
    factor_count: int = 0,
) -> EnvironmentPolicyScheduleContract:
    return EnvironmentPolicyScheduleContractBuilder(
        _market(),
        pre_trade_risk=pre_trade_risk or PreTradeRisk(),
        alpha_enabled=alpha_enabled,
        factor_count=factor_count,
        action_spec=action_spec,
        config=config,
    ).build()


def test_builder_preserves_supplied_config_and_action_spec_identities() -> None:
    config = _config(action_validation_mode=ActionValidationMode.STRICT)
    action_spec = ActionSpec(
        alpha_enabled=True,
        n_factors=2,
        validation_mode=ActionValidationMode.STRICT,
    )

    contract = _build(
        config=config,
        action_spec=action_spec,
        alpha_enabled=True,
        factor_count=2,
    )

    assert contract.config is config
    assert contract.action_spec is action_spec
    assert contract.emergency_risk_monitor.config is config.emergency_risk
    assert contract.action_names == (
        "fast_tilt",
        "slow_tilt",
        "risk_tilt",
        "alpha_scale",
        "factor_0",
        "factor_1",
    )
    assert contract.nominal_episode_bars == 8
    assert contract.nominal_decision_bars == 2
    assert contract.resolved_decision_hours == pytest.approx(2.0)
    assert contract.reward_config == config.resolved_reward_config()


def test_builder_creates_default_action_spec_from_environment_contract() -> None:
    config = _config(action_validation_mode=ActionValidationMode.FAIL_CLOSED)

    contract = _build(
        config=config,
        alpha_enabled=True,
        factor_count=1,
    )

    assert contract.action_spec.alpha_enabled is True
    assert contract.action_spec.n_factors == 1
    assert contract.action_spec.validation_mode is ActionValidationMode.FAIL_CLOSED
    assert contract.action_names[-2:] == ("alpha_scale", "factor_0")


def test_builder_derives_target_weight_names_from_symbols() -> None:
    action_spec = ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        risk_tilt_enabled=False,
        target_weight_count=2,
    )

    contract = _build(config=_config(), action_spec=action_spec)

    assert contract.action_names == ("target_weight:A", "target_weight:B")


def test_builder_uses_configured_decision_hours_without_decision_every() -> None:
    config = _config(
        episode_bars=None,
        decision_every=None,
        episode_hours=8.0,
        decision_hours=2.0,
    )

    contract = _build(config=config)

    assert contract.nominal_episode_bars == 8
    assert contract.nominal_decision_bars == 2
    assert contract.resolved_decision_hours == pytest.approx(2.0)


def test_builder_preserves_default_config_failure() -> None:
    with pytest.raises(
        ValueError,
        match="initial_capital must be explicitly configured in quote-currency units",
    ):
        _build(config=None)


def test_builder_validates_leverage_before_random_initial_gross() -> None:
    pre_trade_risk = PreTradeRisk(PreTradeRiskConfig(max_gross=2.0))
    config = _config(
        random_initial_gross=3.0,
        execution_cost=ExecutionCostConfig(max_leverage=1.0),
    )

    with pytest.raises(
        ValueError,
        match="pre-trade max_gross cannot exceed execution max_leverage",
    ):
        _build(config=config, pre_trade_risk=pre_trade_risk)


def test_builder_rejects_random_initial_gross_after_leverage_passes() -> None:
    pre_trade_risk = PreTradeRisk(PreTradeRiskConfig(max_gross=1.0))

    with pytest.raises(
        ValueError,
        match="random_initial_gross cannot exceed pre-trade max_gross",
    ):
        _build(
            config=_config(random_initial_gross=1.5),
            pre_trade_risk=pre_trade_risk,
        )


def test_builder_validates_alpha_before_factor_count() -> None:
    action_spec = ActionSpec(alpha_enabled=False, n_factors=1)

    with pytest.raises(
        ValueError,
        match="action_spec alpha mode does not match environment",
    ):
        _build(
            config=_config(),
            action_spec=action_spec,
            alpha_enabled=True,
            factor_count=2,
        )


def test_builder_rejects_factor_count_after_alpha_matches() -> None:
    action_spec = ActionSpec(alpha_enabled=True, n_factors=1)

    with pytest.raises(
        ValueError,
        match="action_spec factor count does not match environment",
    ):
        _build(
            config=_config(),
            action_spec=action_spec,
            alpha_enabled=True,
            factor_count=2,
        )


def test_builder_rejects_target_weight_count_after_residual_contracts_match() -> None:
    action_spec = ActionSpec(
        mode=ActionMode.TARGET_WEIGHT,
        risk_tilt_enabled=False,
        target_weight_count=1,
    )

    with pytest.raises(
        ValueError,
        match="target weight count does not match dataset symbols",
    ):
        _build(config=_config(), action_spec=action_spec)


def test_builder_validates_decision_duration_before_episode_choices() -> None:
    config = _config(
        episode_bars=None,
        episode_hours=1.0,
        decision_every=2,
        episode_hour_choices=(0.5,),
    )

    with pytest.raises(
        ValueError,
        match="decision interval cannot exceed episode duration",
    ):
        _build(config=config)


def test_builder_rejects_short_episode_hour_choice_after_duration_passes() -> None:
    config = _config(
        episode_bars=None,
        episode_hours=8.0,
        decision_every=2,
        episode_hour_choices=(1.0,),
    )

    with pytest.raises(
        ValueError,
        match=(
            "episode_hour_choices cannot be shorter than the resolved decision interval"
        ),
    ):
        _build(config=config)


def test_environment_uses_policy_schedule_contract_without_digest_drift() -> None:
    config = _config(action_validation_mode=ActionValidationMode.STRICT)
    action_spec = ActionSpec(validation_mode=ActionValidationMode.STRICT)
    env = ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        action_spec=action_spec,
        config=config,
    )

    assert env.config is config
    assert env.action_spec is action_spec
    assert env.emergency_risk_monitor.config is config.emergency_risk
    assert env.episode_bars == 8
    assert env.decision_bars == 2
    assert env.decision_hours == pytest.approx(2.0)
    assert env._digest_payload()["environment_config"]["resolved_decision_hours"] == 2.0
