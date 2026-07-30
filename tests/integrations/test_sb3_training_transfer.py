from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations import sb3_training
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig

_SOURCE_ENVIRONMENT_DIGEST = "s" * 64
_TARGET_ENVIRONMENT_DIGEST = "t" * 64
_ACTION_NAMES = ("tilt",)
_ACTION_SPEC_DIGEST = content_digest({"names": _ACTION_NAMES})


class TransferTrainingProbe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}
    environment_digest = _TARGET_ENVIRONMENT_DIGEST
    initial_capital = 1_000.0
    decision_hours = 1.0
    action_names = _ACTION_NAMES
    action_spec_digest = _ACTION_SPEC_DIGEST
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
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.close_calls = 0

    @property
    def unwrapped(self) -> TransferTrainingProbe:
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.close_calls += 1


class FakeParameter:
    def numel(self) -> int:
        return 2


class FakePolicy:
    def parameters(self) -> tuple[FakeParameter, ...]:
        return (FakeParameter(),)


class FakeTransferPPO:
    device = "cpu"

    def __init__(self, policy: str, environment: object, **kwargs: object) -> None:
        del policy, environment, kwargs
        self.policy = FakePolicy()
        self.num_timesteps = 0
        self.rollout_buffer_kwargs: dict[str, object] = {}
        self.learn_calls: list[tuple[int, bool]] = []

    def learn(
        self,
        *,
        total_timesteps: int,
        callback: object,
        reset_num_timesteps: bool = True,
    ) -> FakeTransferPPO:
        del callback
        self.learn_calls.append((total_timesteps, reset_num_timesteps))
        self.num_timesteps += total_timesteps
        return self

    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"transferred-policy")


def _config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=1,
        batch_size=1,
        n_epochs=1,
        observation_encoder="flat_mlp",
        device="cpu",
    )


def test_backend_rejects_resume_and_transfer_for_the_same_seed() -> None:
    with pytest.raises(ValueError, match="resume and transfer"):
        StableBaselines3Backend(
            TransferTrainingProbe,
            resume_checkpoint_artifacts={0: Path("resume")},
            transfer_checkpoint_artifacts={0: Path("transfer")},
        )


def test_backend_rejects_replay_resume_with_checkpoint_transfer() -> None:
    with pytest.raises(ValueError, match="replay.*transfer"):
        StableBaselines3Backend(
            TransferTrainingProbe,
            resume_replay_artifact=Path("replay"),
            transfer_checkpoint_artifacts={0: Path("transfer")},
        )


def test_backend_trains_additional_timesteps_after_cross_triplet_transfer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest = SimpleNamespace(
        digest="d" * 64,
        observed_timestep=5,
        environment_digest=_SOURCE_ENVIRONMENT_DIGEST,
    )
    loaded_model = FakeTransferPPO("MlpPolicy", object())
    loaded_model.num_timesteps = 5
    transfer_calls: list[dict[str, object]] = []
    callback_arguments: dict[str, object] = {}

    def load_transfer(**kwargs: object) -> SimpleNamespace:
        transfer_calls.append(dict(kwargs))
        return SimpleNamespace(model=loaded_model, manifest=manifest)

    def build_callback(**kwargs: object) -> object:
        callback_arguments.update(kwargs)
        return object()

    monkeypatch.setattr("stable_baselines3.PPO", FakeTransferPPO)
    monkeypatch.setattr(
        sb3_training,
        "load_sb3_checkpoint_transfer_model",
        load_transfer,
    )
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        build_callback,
    )

    result = StableBaselines3Backend(
        TransferTrainingProbe,
        transfer_checkpoint_artifacts={0: source_root},
    ).train(
        seed=0,
        config=config,
        output_path=tmp_path / "output" / "policy.zip",
    )

    assert len(transfer_calls) == 1
    assert transfer_calls[0]["checkpoint_root"] == source_root
    assert loaded_model.learn_calls == [(2, False)]
    assert result.actual_timesteps == 7
    assert callback_arguments["starting_timestep"] == 5
    assert callback_arguments["total_timesteps"] == 7
    assert not (tmp_path / "output" / "resume.json").exists()

    transfer_payload = json.loads(
        (tmp_path / "output" / "transfer.json").read_text(encoding="utf-8")
    )
    assert transfer_payload == {
        "checkpoint_digest": manifest.digest,
        "checkpoint_observed_timestep": 5,
        "requested_additional_timesteps": 2,
        "schema_version": "training_transfer_v1",
        "source_environment_digest": _SOURCE_ENVIRONMENT_DIGEST,
        "target_environment_digest": _TARGET_ENVIRONMENT_DIGEST,
        "target_total_timesteps": 7,
    }
