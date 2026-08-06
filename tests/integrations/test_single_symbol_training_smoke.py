from __future__ import annotations

import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.sb3_serving import _SB3EnsemblePolicy
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

_ACTION_NAMES = ("target_weight:BTCUSDT",)


class _SingleSymbolEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = "e" * 64
    initial_capital = 100_000.0
    decision_hours = 0.25
    action_names = _ACTION_NAMES
    action_spec_digest = content_digest({"names": _ACTION_NAMES})
    asset_active_column = 1
    layout = ObservationLayout(
        n_symbols=1,
        n_features=1,
        action_size=1,
        n_factors=0,
        per_symbol_width=2,
        global_width=1,
    )

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self._step = 0
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert np.asarray(action).shape == (1,)
        self._step += 1
        phase = self._step / 8.0
        observation = np.asarray(
            (math.sin(phase), 1.0, math.cos(phase)), dtype=np.float32
        )
        reward = -float(np.square(action - math.sin(phase * 0.5)).sum())
        return observation, reward, self._step >= 8, False, {}


def _training_config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0,),
        algorithm="ppo",
        batch_size=8,
        learning_rate=3e-4,
        policy_net_arch=(16, 8),
        value_net_arch=(16, 8),
        checkpoint_interval_steps=8,
        max_checkpoints=1,
        device="cpu",
        n_steps=8,
        n_epochs=1,
        observation_encoder="flat_mlp",
    )


def test_one_action_ppo_publishes_reloadable_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    output = tmp_path / "policy.zip"
    config = _training_config()

    result = StableBaselines3Backend(_SingleSymbolEnv).train(
        seed=0,
        config=config,
        output_path=output,
    )

    assert result.actual_timesteps == 8
    assert output.is_file()
    assert tuple(result.action_names) == _ACTION_NAMES
    checkpoints = tuple((tmp_path / "checkpoints").glob("step-*/checkpoint.json"))
    assert len(checkpoints) == 1
    model = PPO.load(str(output), device="cpu")
    assert model.action_space.shape == (1,)
    action, _ = model.predict(
        np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
        deterministic=True,
    )
    assert np.asarray(action).reshape(-1).shape == (1,)


def test_one_symbol_oracle_bc_runs_before_ppo(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    n_bars = 80
    close = np.linspace(100.0, 130.0, n_bars, dtype=np.float64)[:, None]
    dataset = MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )

    def factory() -> ResidualMarketEnv:
        return ResidualMarketEnv(
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
                initial_capital=100_000.0,
                episode_bars=8,
                decision_every=1,
                execution_cost=ExecutionCostConfig.zero(),
            ),
        )

    result = StableBaselines3Backend(factory).train(
        seed=3,
        config=ResidualTrainingConfig(
            timesteps=4,
            gamma=0.99,
            seeds=(3,),
            n_steps=2,
            n_envs=2,
            batch_size=4,
            n_epochs=1,
            observation_encoder="flat_mlp",
            device="cpu",
            behavior_cloning_epochs=15,
            behavior_cloning_batch_size=16,
            behavior_cloning_validation_fraction=0.1,
        ),
        output_path=tmp_path / "member" / "policy.zip",
    )

    assert result.actual_timesteps == 4
    assert result.action_size == 1
    assert result.action_names == _ACTION_NAMES
    assert (tmp_path / "member" / "teacher" / "manifest.json").is_file()
    assert (tmp_path / "member" / "behavior-cloning.json").is_file()
    assert (tmp_path / "member" / "oracle-evaluation.json").is_file()
    cloning = json.loads(
        (tmp_path / "member" / "behavior-cloning.json").read_text(encoding="utf-8")
    )
    assert cloning["oracle_reproduction"]["passed"] is True


class _Predictor:
    def __init__(self, action: tuple[float, ...]) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[np.ndarray, None]:
        assert deterministic is True
        assert observation.shape == (3,)
        return self.action.copy(), None


def test_one_action_serving_ensemble_is_fail_closed() -> None:
    policy = _SB3EnsemblePolicy(
        (_Predictor((0.25,)), _Predictor((0.75,))),
        observation_size=3,
        action_size=1,
    )

    action = policy.predict(np.asarray((0.0, 1.0, 0.0), dtype=np.float32))

    np.testing.assert_allclose(action, np.asarray((0.5,), dtype=np.float32))
    assert action.shape == (1,)

    invalid = _SB3EnsemblePolicy(
        (_Predictor((0.1, 0.2, 0.3)),),
        observation_size=3,
        action_size=1,
    )
    with pytest.raises(ValueError, match="action shape mismatch"):
        invalid.predict(np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
