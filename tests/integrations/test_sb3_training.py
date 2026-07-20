from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations import sb3_training
from trade_rl.integrations.sb3_training import (
    StableBaselines3Backend,
    _build_training_environment,
)
from trade_rl.learning import OracleTeacherConfig
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

ENVIRONMENT_DIGEST = "e" * 64
ACTION_NAMES = ("tilt",)
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


class TinyEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

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


def test_build_training_environment_explicitly_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stable_baselines3.common import vec_env

    observed: dict[str, object] = {}

    class FakeSubprocVecEnv:
        def __init__(
            self,
            factories: list[Callable[[], TinyEnvironment]],
            start_method: str | None = None,
        ) -> None:
            observed["factory_count"] = len(factories)
            observed["start_method"] = start_method

    monkeypatch.setattr(vec_env, "SubprocVecEnv", FakeSubprocVecEnv)

    environment = _build_training_environment(_tiny_environment_factory, 2)

    assert isinstance(environment, FakeSubprocVecEnv)
    assert observed == {"factory_count": 2, "start_method": "spawn"}


def test_build_training_environment_uses_in_process_workers_for_sequences() -> None:
    factory: Callable[[], TinyEnvironment] = _tiny_environment_factory
    environment = _build_training_environment(factory, 2, subprocesses=False)
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv

        assert isinstance(environment, DummyVecEnv)
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

    def build_workers(
        worker_factory: Callable[[], Any], n_envs: int, *, subprocesses: bool = True
    ) -> Any:
        assert worker_factory is factory
        assert n_envs == 2
        assert subprocesses is False
        assert events == ["metadata", "validated", "metadata", "probe-close"]
        events.append("workers-build")
        return vector_environment

    def validate_probe(
        identity: dict[str, Any], config: ResidualTrainingConfig
    ) -> None:
        validate_environment(identity, config)
        events.append("validated")

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        action_distribution_name = "squashed_diag_gaussian"

        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)

    class FakePPO:
        device = "cpu"
        num_timesteps = 0

        def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
            assert environment is vector_environment
            self.policy = FakePolicy()
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
    architecture = json.loads(
        (tmp_path / "model-architecture.json").read_text(encoding="utf-8")
    )
    assert architecture["architecture"].get("action_distribution") == (
        "squashed_diag_gaussian"
    )
    assert factory_calls == 1
    assert probe.close_calls == 1
    assert vector_environment.close_calls == 1
    assert events[-2:] == ["workers-build", "vector-close"]


def test_backend_runs_oracle_behavior_cloning_before_ppo(tmp_path: Path) -> None:
    n_bars = 40
    close = np.column_stack(
        [
            np.linspace(100.0, 130.0, n_bars),
            np.linspace(100.0, 80.0, n_bars),
        ]
    )
    dataset = MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTC", "ETH"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
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
                target_weight_count=2,
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
            timesteps=2,
            gamma=0.99,
            seeds=(3,),
            n_steps=2,
            n_envs=1,
            batch_size=2,
            n_epochs=1,
            asset_set_encoder=False,
            device="cpu",
            behavior_cloning_epochs=1,
            behavior_cloning_batch_size=16,
        ),
        output_path=tmp_path / "member" / "policy.zip",
    )

    assert result.actual_timesteps == 2
    assert (tmp_path / "member" / "teacher" / "manifest.json").is_file()
    assert (tmp_path / "member" / "behavior-cloning.json").is_file()


def test_backend_caches_oracle_targets_across_seed_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = type("Dataset", (), {"dataset_id": "d" * 64})()
    config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    calls = 0

    def calculate(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.asarray([[0.0], [0.25]], dtype=np.float32)

    monkeypatch.setattr(sb3_training, "oracle_target_path", calculate)
    backend = StableBaselines3Backend(_tiny_environment_factory)

    first = backend._oracle_targets(dataset, (3, 6), config)
    second = backend._oracle_targets(dataset, (3, 6), config)

    assert calls == 1
    assert first is second
    assert first.flags.writeable is False


def test_backend_caches_teacher_dataset_across_seed_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset

    calls = 0

    def collect(
        environment: Any,
        targets: np.ndarray,
        *,
        dataset_id: str,
        train_range: tuple[int, int],
        teacher_config_digest: str,
    ) -> SupervisedPolicyDataset:
        nonlocal calls
        calls += 1
        start, stop = train_range
        return SupervisedPolicyDataset(
            observations=np.zeros((stop - start - 1, 2), dtype=np.float32),
            actions=np.asarray(targets, dtype=np.float32),
            dataset_id=dataset_id,
            train_start=start,
            train_stop=stop,
            environment_digest=environment.environment_digest,
            action_spec_digest=environment.action_spec_digest,
            teacher_config_digest=teacher_config_digest,
        )

    monkeypatch.setattr(sb3_training, "collect_teacher_rollout", collect)
    backend = StableBaselines3Backend(_tiny_environment_factory)
    teacher_config = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    targets = np.asarray([[0.0], [0.25]], dtype=np.float32)
    environment = type(
        "TeacherEnvironment",
        (),
        {
            "environment_digest": "1" * 64,
            "action_spec_digest": "2" * 64,
        },
    )()

    first = backend._teacher_dataset(
        environment,
        targets,
        dataset_id="3" * 64,
        train_range=(3, 6),
        teacher_config=teacher_config,
    )
    second = backend._teacher_dataset(
        environment,
        targets,
        dataset_id="3" * 64,
        train_range=(3, 6),
        teacher_config=teacher_config,
    )
    changed_environment = type(
        "TeacherEnvironment",
        (),
        {
            "environment_digest": "4" * 64,
            "action_spec_digest": "2" * 64,
        },
    )()
    third = backend._teacher_dataset(
        changed_environment,
        targets,
        dataset_id="3" * 64,
        train_range=(3, 6),
        teacher_config=teacher_config,
    )

    assert calls == 2
    assert first is second
    assert third is not first


def test_backend_rejects_ppo_rollout_before_worker_or_model_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe = TrainingProbe([])
    model_created = False

    class ForbiddenPPO:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal model_created
            model_created = True

    monkeypatch.setattr("stable_baselines3.PPO", ForbiddenPPO)
    config = replace(_training_config(), max_rollout_buffer_bytes=1)
    with pytest.raises(ValueError, match="rollout buffer"):
        StableBaselines3Backend(lambda: probe).train(
            seed=0,
            config=config,
            output_path=tmp_path / "policy.zip",
        )
    assert model_created is False
    assert probe.close_calls == 1


def test_backend_resumes_ppo_checkpoint_to_requested_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.rl.checkpointing import publish_checkpoint

    config = ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=1,
        batch_size=1,
        n_epochs=1,
        asset_set_encoder=False,
        device="cpu",
    )

    class CheckpointSource:
        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resume-policy")

    manifest = publish_checkpoint(
        model=CheckpointSource(),
        checkpoint_root=tmp_path / "resume",
        algorithm="ppo",
        seed=0,
        requested_timestep=1,
        observed_timestep=1,
        environment_digest=ENVIRONMENT_DIGEST,
        training_config_digest=content_digest(config.digest_payload()),
    )
    events: list[object] = []

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        def parameters(self):
            return (FakeParameter(),)

    class FakeResumePPO:
        device = "cpu"

        def __init__(self, policy, environment, **kwargs):
            self.policy = FakePolicy()
            self.num_timesteps = 0
            self.rollout_buffer_kwargs = {}

        @classmethod
        def load(cls, path, env=None, device=None):
            events.append(("load", Path(path), device, env is not None))
            model = cls("MlpPolicy", env)
            model.num_timesteps = 1
            return model

        def learn(self, *, total_timesteps, callback, reset_num_timesteps=True):
            events.append(("learn", total_timesteps, reset_num_timesteps))
            self.num_timesteps += total_timesteps
            return self

        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resumed-policy")

    monkeypatch.setattr("stable_baselines3.PPO", FakeResumePPO)
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )
    result = StableBaselines3Backend(
        lambda: TrainingProbe([]),
        resume_checkpoint_artifacts={0: manifest.policy_path.parent},
    ).train(
        seed=0,
        config=config,
        output_path=tmp_path / "output" / "policy.zip",
    )
    assert result.actual_timesteps == 2
    assert events[0][0] == "load"
    assert ("learn", 1, False) in events
    resume_payload = (tmp_path / "output" / "resume.json").read_text(encoding="utf-8")
    assert manifest.digest in resume_payload
