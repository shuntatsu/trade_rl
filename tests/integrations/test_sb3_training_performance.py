from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.training import ResidualTrainingConfig


class _Probe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = "e" * 64
    initial_capital = 1_000.0
    decision_hours = 1.0
    action_names = ("target_weight:BTC",)
    action_spec_digest = "a" * 64
    observation_schema = "performance_probe_v1"
    observation_contract_digest = "b" * 64
    alpha_artifact_digest = None
    factor_artifact_digest = None
    normalizer = None
    sequence_normalizer = None

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


class _Parameter:
    def numel(self) -> int:
        return 2


class _Policy:
    action_distribution_name = "squashed_diag_gaussian"

    def parameters(self) -> tuple[_Parameter, ...]:
        return (_Parameter(),)

    def extract_features(self, observations: object) -> object:
        return observations


class _FakePPO:
    device = "cpu"
    num_timesteps = 0

    def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
        self.policy = _Policy()
        self.env = environment

    def collect_rollouts(self) -> bool:
        return True

    def train(self) -> None:
        return None

    def learn(self, *, total_timesteps: int, callback: Any, **kwargs: Any) -> None:
        self.collect_rollouts()
        self.train()
        self.policy.extract_features(np.zeros((1, 2), dtype=np.float32))
        self.env.step(np.zeros((1,), dtype=np.float32))
        self.num_timesteps += total_timesteps

    def save(self, target: str) -> None:
        Path(f"{target}.zip").write_bytes(b"policy")


def test_backend_writes_member_training_performance_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("stable_baselines3.PPO", _FakePPO)
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    result = StableBaselines3Backend(_Probe).train(
        seed=0,
        config=ResidualTrainingConfig(
            timesteps=2,
            gamma=0.99,
            seeds=(0,),
            n_steps=1,
            n_envs=1,
            batch_size=1,
            n_epochs=1,
            observation_encoder=("flat_mlp"),
            device="cpu",
        ),
        output_path=tmp_path / "policy.zip",
    )

    assert result.actual_timesteps == 2
    path = tmp_path / "training-performance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("digest")
    assert payload["schema_version"] == "training_performance_evidence_v1"
    assert payload["device_type"] == "cpu"
    assert payload["requested_environment_steps"] == 2
    assert payload["observed_environment_steps"] == 2
    assert payload["collect_rollouts_calls"] == 1
    assert payload["optimization_calls"] == 1
    assert payload["feature_extraction_calls"] == 1
    assert payload["environment_step_calls"] == 1
    assert payload["peak_cuda_allocated_bytes"] is None
    assert payload["peak_cuda_reserved_bytes"] is None
    assert digest == content_digest(payload)
    assert len(digest) == 64
