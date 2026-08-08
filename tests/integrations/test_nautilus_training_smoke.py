from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")
pytest.importorskip("stable_baselines3")

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.rl_dual_shadow import NautilusEnvironmentDualShadow
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.dual_shadow_environment import ExecutionDualShadowResidualMarketEnv
from trade_rl.rl.environment import ResidualMarketEnvConfig
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _market() -> MarketDataset:
    n_bars = 80
    close = np.linspace(100.0, 101.0, n_bars, dtype=np.float64)[:, None]
    return MarketDataset(
        dataset_id="9" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
        index_price=close.copy(),
    )


def _factory(dataset: MarketDataset):
    def build() -> ExecutionDualShadowResidualMarketEnv:
        return ExecutionDualShadowResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(
                TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
            ),
            action_spec=ActionSpec(
                mode="target_weight",
                risk_tilt_enabled=False,
                target_weight_count=1,
            ),
            config=ResidualMarketEnvConfig(
                initial_capital=1_000.0,
                episode_bars=3,
                decision_every=1,
                execution_cost=ExecutionCostConfig.zero(),
                initial_state_modes=("cash",),
            ),
            execution_dual_shadow=NautilusEnvironmentDualShadow(
                dataset,
                no_trade_band=0.0,
            ),
        )

    return build


def _ppo_config(*, algorithm: str) -> ResidualTrainingConfig:
    common = dict(
        timesteps=3,
        gamma=0.99,
        seeds=(0,),
        algorithm=algorithm,
        n_steps=3,
        n_envs=1,
        batch_size=3,
        n_epochs=1,
        observation_encoder="flat_mlp",
        device="cpu",
        policy_net_arch=(16, 8),
        value_net_arch=(16, 8),
    )
    if algorithm == "ppo":
        return ResidualTrainingConfig(**common)
    count = len(CONSTRAINT_COST_NAMES)
    return ResidualTrainingConfig(
        **common,
        lagrangian_budgets=(0.1,) * count,
        lagrangian_dual_learning_rates=(0.05,) * count,
        lagrangian_ema_betas=(0.9,) * count,
        lagrangian_initial_multipliers=(0.0,) * count,
        lagrangian_max_multipliers=(10.0,) * count,
        lagrangian_warmup_rollouts=(0,) * count,
        lagrangian_update_interval_rollouts=(1,) * count,
        lagrangian_minimum_completed_episodes=(1,) * count,
        lagrangian_probe_episodes=1,
        lagrangian_probe_max_steps_per_episode=3,
    )


@pytest.mark.nautilus
@pytest.mark.parametrize("algorithm", ("ppo", "lagrangian_ppo"))
def test_three_step_training_smoke_uses_nautilus_dual_shadow_runtime(
    algorithm: str,
    tmp_path: Path,
) -> None:
    dataset = _market()

    result = StableBaselines3Backend(_factory(dataset)).train(
        seed=0,
        config=_ppo_config(algorithm=algorithm),
        output_path=tmp_path / algorithm / "policy.zip",
    )

    assert result.actual_timesteps == 3
    assert result.action_size == 1
    assert result.action_names == ("target_weight:BTCUSDT",)
    assert (tmp_path / algorithm / "policy.zip").is_file()
