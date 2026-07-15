from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations import sb3_training
from trade_rl.integrations.sb3_training import (
    StableBaselines3Backend,
    _build_training_environment,
)
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig

ENVIRONMENT_DIGEST = "e" * 64
ACTION_NAMES = ("tilt",)
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


class TinyEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
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


def _tiny_environment_factory() -> TinyEnvironment:
    return TinyEnvironment()


class TrainingProbe(TinyEnvironment):
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

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.close_calls = 0

    @property
    def unwrapped(self) -> TrainingProbe:
        self.events.append("metadata")
        return self

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("probe-close")


class RaisingCloseProbe(TrainingProbe):
    def close(self) -> None:
        super().close()
        raise RuntimeError("probe close failed")


class VectorEnvironment:
    num_envs = 2

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("vector-close")


def _training_config(*, asset_set_encoder: bool = False) -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=2,
        batch_size=2,
        n_epochs=1,
        asset_set_encoder=asset_set_encoder,
        device="cpu",
    )


def test_build_training_environment_returns_direct_environment_for_width_one() -> None:
    calls = 0

    def factory() -> TinyEnvironment:
        nonlocal calls
        calls += 1
        return TinyEnvironment()

    environment = _build_training_environment(factory, 1)
    try:
        assert isinstance(environment, TinyEnvironment)
        assert calls == 1
    finally:
        environment.close()


def test_build_training_environment_uses_two_subprocess_workers() -> None:
    factory: Callable[[], TinyEnvironment] = _tiny_environment_factory
    environment = _build_training_environment(factory, 2)
    try:
        assert environment.num_envs == 2
    finally:
        environment.close()


def test_backend_closes_a_failing_probe_exactly_once(tmp_path: Path) -> None:
    probe = RaisingCloseProbe([])
    backend = StableBaselines3Backend(lambda: probe)

    with pytest.raises(RuntimeError, match="probe close failed"):
        backend.train(
            seed=0,
            config=_training_config(),
            output_path=tmp_path / "policy.zip",
        )

    assert probe.close_calls == 1


def test_backend_builds_workers_after_probe_validation_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    probe = TrainingProbe(events)
    vector_environment = VectorEnvironment(events)
    factory_calls = 0
    model_arguments: dict[str, Any] = {}
    validate_environment = sb3_training._validate_training_environment

    def factory() -> TrainingProbe:
        nonlocal factory_calls
        factory_calls += 1
        return probe

    def build_workers(worker_factory: Callable[[], Any], n_envs: int) -> Any:
        assert worker_factory is factory
        assert n_envs == 2
        assert events == ["metadata", "validated", "metadata", "probe-close"]
        events.append("workers-build")
        return vector_environment

    def validate_probe(
        identity: dict[str, Any], config: ResidualTrainingConfig
    ) -> None:
        validate_environment(identity, config)
        events.append("validated")

    class FakePPO:
        device = "cpu"
        num_timesteps = 0

        def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
            assert environment is vector_environment
            model_arguments.update({"policy": policy, **kwargs})

        def learn(self, *, total_timesteps: int, callback: Any) -> None:
            self.num_timesteps = total_timesteps

        def save(self, target: str) -> None:
            Path(f"{target}.zip").write_bytes(b"policy")

    monkeypatch.setattr(sb3_training, "_build_training_environment", build_workers)
    monkeypatch.setattr(sb3_training, "_validate_training_environment", validate_probe)
    monkeypatch.setattr("stable_baselines3.PPO", FakePPO)
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )

    result = StableBaselines3Backend(factory).train(
        seed=0,
        config=_training_config(asset_set_encoder=True),
        output_path=tmp_path / "policy.zip",
    )

    extractor = model_arguments["policy_kwargs"]["features_extractor_kwargs"]
    assert extractor == {
        "n_symbols": 1,
        "per_symbol_width": 2,
        "global_width": 0,
        "active_column": 1,
        "asset_embedding_dim": 64,
        "global_embedding_dim": 64,
    }
    assert result.actual_timesteps == 2
    assert factory_calls == 1
    assert probe.close_calls == 1
    assert vector_environment.close_calls == 1
    assert events[-2:] == ["workers-build", "vector-close"]
