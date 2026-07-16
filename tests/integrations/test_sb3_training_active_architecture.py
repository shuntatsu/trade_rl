from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig

ENVIRONMENT_DIGEST = "e" * 64
ACTION_NAMES = ("tilt",)
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


class TrainingProbe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = ENVIRONMENT_DIGEST
    initial_capital = 1_000.0
    decision_hours = 1.0
    action_names = ACTION_NAMES
    action_spec_digest = ACTION_SPEC_DIGEST
    asset_active_column = 1
    layout = ObservationLayout(
        n_symbols=1,
        n_features=1,
        action_size=1,
        n_factors=0,
        per_symbol_width=2,
        global_width=0,
    )

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


def test_off_policy_backend_passes_distinct_actor_and_critic_architectures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_arguments: dict[str, Any] = {}

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)

    class FakeSAC:
        device = "cpu"
        num_timesteps = 0

        def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
            self.policy = FakePolicy()
            model_arguments.update({"policy": policy, **kwargs})

        def learn(self, **kwargs: Any) -> FakeSAC:
            self.num_timesteps += int(kwargs["total_timesteps"])
            return self

        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"policy")

        def save_replay_buffer(self, target: str) -> None:
            Path(target).write_bytes(b"replay")

    monkeypatch.setattr("stable_baselines3.SAC", FakeSAC)
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    config = ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        algorithm="sac",
        batch_size=1,
        buffer_size=8,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        policy_net_arch=(16, 8),
        value_net_arch=(32, 24),
        asset_set_encoder=False,
        device="cpu",
    )
    StableBaselines3Backend(TrainingProbe).train(
        seed=0,
        config=config,
        output_path=tmp_path / "policy.zip",
    )

    assert model_arguments["policy_kwargs"]["net_arch"] == {
        "pi": [16, 8],
        "qf": [32, 24],
    }
