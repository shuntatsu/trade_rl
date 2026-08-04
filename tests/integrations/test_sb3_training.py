from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
    _behavior_cloning_quality,
    _build_training_environment,
    _compact_training_info,
    _configure_torch_cuda_runtime,
    _oracle_solver_config,
    _teacher_worker_count,
)
from trade_rl.learning import OracleTeacherConfig
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.training_modes import CudaRuntimeMode
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

ENVIRONMENT_DIGEST = "e" * 64
ACTION_NAMES = ("tilt",)
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


class _BackendFlag:
    def __init__(self, value: bool) -> None:
        self.allow_tf32 = value


class _CudnnFlags(_BackendFlag):
    def __init__(self) -> None:
        super().__init__(True)
        self.benchmark = False
        self.deterministic = True


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, bf16_supported: bool = True) -> None:
        self.cuda = type(
            "Cuda",
            (),
            {
                "is_available": lambda _: cuda_available,
                "is_bf16_supported": lambda _: bf16_supported,
            },
        )()
        self.backends = type(
            "Backends",
            (),
            {
                "cuda": type("CudaBackend", (), {"matmul": _BackendFlag(False)})(),
                "cudnn": _CudnnFlags(),
            },
        )()
        self.precision = "highest"
        self.deterministic_algorithms = False

    def device(self, value: object) -> object:
        kind = str(value).split(":", maxsplit=1)[0]
        if kind not in {"cpu", "cuda"}:
            raise RuntimeError("invalid device")
        return type("Device", (), {"type": kind})()

    def get_float32_matmul_precision(self) -> str:
        return self.precision

    def set_float32_matmul_precision(self, value: str) -> None:
        self.precision = value
        self.backends.cuda.matmul.allow_tf32 = value == "high"

    def use_deterministic_algorithms(
        self, enabled: bool, *, warn_only: bool = False
    ) -> None:
        del warn_only
        self.deterministic_algorithms = enabled

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic_algorithms


def test_cuda_runtime_enables_tf32_and_fixed_shape_cudnn_search() -> None:
    torch = _FakeTorch(cuda_available=True)

    result = _configure_torch_cuda_runtime(torch, "cuda:0", CudaRuntimeMode.PERFORMANCE)

    assert result == {
        "mode": "performance",
        "deterministic_algorithms": False,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "cudnn_tf32": True,
        "float32_matmul_precision": "high",
        "matmul_tf32": True,
        "sequence_encoder_autocast": "bfloat16",
    }


def test_cpu_runtime_does_not_enable_cuda_fast_paths() -> None:
    torch = _FakeTorch(cuda_available=True)

    result = _configure_torch_cuda_runtime(torch, "cpu", CudaRuntimeMode.PERFORMANCE)

    assert result["cudnn_benchmark"] is False
    assert result["cudnn_deterministic"] is True
    assert result["matmul_tf32"] is False
    assert result["float32_matmul_precision"] == "highest"
    assert result["sequence_encoder_autocast"] == "disabled"


def test_cuda_runtime_deterministic_mode_disables_fast_nondeterministic_paths() -> None:
    torch = _FakeTorch(cuda_available=True)

    result = _configure_torch_cuda_runtime(
        torch, "cuda:0", CudaRuntimeMode.DETERMINISTIC
    )

    assert result["mode"] == "deterministic"
    assert result["deterministic_algorithms"] is True
    assert result["cudnn_benchmark"] is False
    assert result["cudnn_deterministic"] is True
    assert result["matmul_tf32"] is False
    assert result["cudnn_tf32"] is False
    assert result["float32_matmul_precision"] == "highest"


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


def _training_config(
    *, observation_encoder: str = "flat_mlp"
) -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=2,
        batch_size=2,
        n_epochs=1,
        observation_encoder=observation_encoder,
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


def test_compact_training_info_removes_history_bearing_execution_results() -> None:
    history = list(np.linspace(-0.01, 0.01, 10_000))
    execution = SimpleNamespace(
        book=SimpleNamespace(
            weights=np.asarray((0.25, -0.5), dtype=np.float64),
            returns_history=history,
        )
    )
    info: dict[str, object] = {
        "hybrid_execution": execution,
        "shadow_execution": execution,
        "hybrid_liquidation": execution,
        "shadow_liquidation": execution,
        "hybrid_risk": SimpleNamespace(reasons=("drawdown_deleveraging",)),
        "portfolio_value_after": 99_000.0,
    }

    compact = _compact_training_info(info)

    assert all(
        key not in compact
        for key in (
            "hybrid_execution",
            "shadow_execution",
            "hybrid_liquidation",
            "shadow_liquidation",
        )
    )
    assert compact["telemetry_weights_after"] == pytest.approx((0.25, -0.5))
    assert compact["telemetry_risk_reasons"] == ("drawdown_deleveraging",)
    assert compact["portfolio_value_after"] == 99_000.0
    assert info["hybrid_execution"] is execution
    assert execution.book.returns_history is history


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
        assert worker_factory is not factory
        assert callable(worker_factory)
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
        config=_training_config(observation_encoder=("asset_set")),
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
    # Keep the chronological holdout large enough to make the Oracle
    # reproduction gate meaningful instead of hinging on only four actions.
    n_bars = 80
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
            timesteps=4,
            gamma=0.99,
            seeds=(3,),
            n_steps=2,
            n_envs=2,
            batch_size=4,
            n_epochs=1,
            observation_encoder=("flat_mlp"),
            device="cpu",
            behavior_cloning_epochs=15,
            behavior_cloning_batch_size=16,
            behavior_cloning_validation_fraction=0.1,
        ),
        output_path=tmp_path / "member" / "policy.zip",
    )

    assert result.actual_timesteps == 4
    assert (tmp_path / "member" / "teacher" / "manifest.json").is_file()
    assert (tmp_path / "member" / "behavior-cloning.json").is_file()
    assert (tmp_path / "member" / "oracle-evaluation.json").is_file()
    assert (tmp_path / "member" / "behavior-cloning-holdout.json").is_file()
    cloning = json.loads(
        (tmp_path / "member" / "behavior-cloning.json").read_text(encoding="utf-8")
    )
    assert cloning["oracle_reproduction"]["passed"] is True
    assert cloning["oracle_reproduction"]["required"] is False


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


def test_behavior_cloning_quality_gate_uses_relative_mse_improvement() -> None:
    relative, passed = _behavior_cloning_quality(
        initial_mse=0.12,
        final_mse=0.11,
        required_relative_improvement=0.05,
    )
    assert relative == pytest.approx(1.0 / 12.0)
    assert passed is True

    relative, passed = _behavior_cloning_quality(
        initial_mse=0.12,
        final_mse=0.119,
        required_relative_improvement=0.05,
    )
    assert relative == pytest.approx(1.0 / 120.0)
    assert passed is False


def test_backend_caches_causal_trend_targets_without_using_stop_bar() -> None:
    dataset = type("Dataset", (), {"dataset_id": "d" * 64})()
    calls: list[int] = []

    class Strategy:
        def targets(self, observed_dataset: Any, index: int) -> SimpleNamespace:
            assert observed_dataset is dataset
            calls.append(index)
            return SimpleNamespace(base=np.asarray([index, -index], dtype=np.float32))

    backend = StableBaselines3Backend(_tiny_environment_factory)
    first = backend._trend_baseline_targets(
        dataset,
        (3, 7),
        Strategy(),
        teacher_digest="a" * 64,
    )
    second = backend._trend_baseline_targets(
        dataset,
        (3, 7),
        Strategy(),
        teacher_digest="a" * 64,
    )

    assert calls == [3, 4, 5]
    np.testing.assert_array_equal(
        first,
        np.asarray([[3, -3], [4, -4], [5, -5]], dtype=np.float32),
    )
    assert first.flags.writeable is False
    assert second is first


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


def test_backend_reuses_identity_bound_teacher_artifact_across_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    cache_root = tmp_path / "teacher-cache"
    monkeypatch.setenv("TRADE_RL_TEACHER_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(sb3_training, "collect_teacher_rollout", collect)
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
    arguments = {
        "dataset_id": "3" * 64,
        "train_range": (3, 6),
        "teacher_config": teacher_config,
    }

    first = StableBaselines3Backend(_tiny_environment_factory)._teacher_dataset(
        environment, targets, **arguments
    )
    second = StableBaselines3Backend(_tiny_environment_factory)._teacher_dataset(
        environment, targets, **arguments
    )

    assert calls == 1
    assert first.observation_digest == second.observation_digest
    assert first.action_digest == second.action_digest
    cache_entries = tuple(cache_root.iterdir())
    assert len(cache_entries) == 1
    assert {path.name for path in cache_entries[0].iterdir()} == {
        "arrays.npz",
        "manifest.json",
    }


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
    from trade_rl.rl.policy_identity import bind_sb3_policy_identity

    config = ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=1,
        batch_size=1,
        n_epochs=1,
        observation_encoder=("flat_mlp"),
        device="cpu",
    )

    class CheckpointSource:
        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resume-policy")

    checkpoint_source = CheckpointSource()
    bind_sb3_policy_identity(
        checkpoint_source,
        SimpleNamespace(
            observation_encoder="flat_mlp",
            sequence_symbols=None,
            sequence_action_names=None,
        ),
    )
    manifest = publish_checkpoint(
        model=checkpoint_source,
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


def test_backend_wires_learning_rate_schedule_and_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    events: list[str] = []
    vector_environment = VectorEnvironment(events)

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        action_distribution_name = "squashed_diag_gaussian"

        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)

    class CapturingPPO:
        device = "cpu"
        num_timesteps = 0

        def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
            assert environment is vector_environment
            self.policy = FakePolicy()
            captured["constructor"] = {"policy": policy, **kwargs}

        def learn(self, **kwargs: Any) -> "CapturingPPO":
            captured["learn"] = kwargs
            self.num_timesteps = int(kwargs["total_timesteps"])
            return self

        def save(self, target: str) -> None:
            Path(f"{target}.zip").write_bytes(b"policy")

    monkeypatch.setattr("stable_baselines3.PPO", CapturingPPO)
    monkeypatch.setattr(
        sb3_training,
        "_build_training_environment",
        lambda *args, **kwargs: vector_environment,
    )
    backend = StableBaselines3Backend(lambda: TrainingProbe(events))
    config = replace(
        _training_config(),
        learning_rate_schedule="linear",
        tensorboard_enabled=True,
    )

    backend.train(
        seed=0,
        config=config,
        output_path=tmp_path / "member" / "policy.zip",
    )

    constructor = captured["constructor"]
    assert isinstance(constructor, dict)
    assert callable(constructor["learning_rate"])
    assert constructor["tensorboard_log"] == str(tmp_path / "member" / "tensorboard")
    learn = captured["learn"]
    assert isinstance(learn, dict)
    assert learn["tb_log_name"] == "seed-0-ppo"


def test_hierarchical_teacher_labels_bind_effective_current_weights() -> None:
    from trade_rl.integrations.sb3_training import _hierarchical_teacher_labels
    from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset

    class HierarchicalPolicy:
        def hierarchical_actor_outputs(self) -> None:
            return None

    observations = {
        "current_weights": np.array([[0.0], [0.4], [0.4]], dtype=np.float32),
        "active": np.ones((3, 1), dtype=np.float32),
    }
    dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=np.array([[0.4], [0.4], [0.0]], dtype=np.float32),
        dataset_id="1" * 64,
        train_start=0,
        train_stop=4,
        environment_digest="2" * 64,
        action_spec_digest="3" * 64,
        teacher_config_digest="4" * 64,
    )
    config = SimpleNamespace(behavior_cloning_gate_change_threshold=0.05)

    labels = _hierarchical_teacher_labels(
        policy=HierarchicalPolicy(),
        teacher_dataset=dataset,
        config=config,
    )

    assert labels is not None
    assert labels.gate_labels[:, 0].tolist() == [True, False, True]
    np.testing.assert_array_equal(
        labels.current_weights, observations["current_weights"]
    )
    assert labels.source_teacher_digest == dataset.action_digest


def test_hierarchical_teacher_labels_fail_closed_without_v3_threshold() -> None:
    from trade_rl.integrations.sb3_training import _hierarchical_teacher_labels
    from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset

    class HierarchicalPolicy:
        def hierarchical_actor_outputs(self) -> None:
            return None

    dataset = SupervisedPolicyDataset(
        observations={
            "current_weights": np.zeros((2, 1), dtype=np.float32),
            "active": np.ones((2, 1), dtype=np.float32),
        },
        actions=np.array([[0.2], [0.0]], dtype=np.float32),
        dataset_id="1" * 64,
        train_start=0,
        train_stop=3,
        environment_digest="2" * 64,
        action_spec_digest="3" * 64,
        teacher_config_digest="4" * 64,
    )

    with pytest.raises(ValueError, match="training_run_config_v4"):
        _hierarchical_teacher_labels(
            policy=HierarchicalPolicy(),
            teacher_dataset=dataset,
            config=SimpleNamespace(),
        )


def test_bc_gate_enforcement_rejects_zero_trade_report() -> None:
    from trade_rl.integrations.sb3_training import _enforce_behavior_cloning_gates
    from trade_rl.learning.evaluation import (
        BehaviorCloningGateEvaluation,
        BehaviorCloningGateGroup,
        BehaviorCloningGateMetric,
    )

    passed = BehaviorCloningGateMetric(
        name="gate_recall",
        status="passed",
        observed=0.8,
        comparison=">=",
        threshold=0.6,
        support=8,
        minimum_support=1,
        reason="gate_recall passed",
    )
    zero_trade = BehaviorCloningGateMetric(
        name="executed_change_count",
        status="failed",
        observed=0,
        comparison=">=",
        threshold=1,
        support=8,
        minimum_support=1,
        reason="zero-trade collapse: causal holdout executed no target changes",
    )
    report = BehaviorCloningGateEvaluation(
        teacher_reconstruction_gate=BehaviorCloningGateGroup(
            name="teacher_reconstruction_gate",
            metrics=(passed,),
        ),
        causal_non_collapse_gate=BehaviorCloningGateGroup(
            name="causal_non_collapse_gate",
            metrics=(zero_trade,),
        ),
    )

    with pytest.raises(RuntimeError, match="zero-trade collapse"):
        _enforce_behavior_cloning_gates(report)


_ORACLE_ENV_KEYS = (
    "TRADE_RL_ORACLE_SOLVER",
    "TRADE_RL_ORACLE_EPISODE_BATCH_SIZE",
    "TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE",
    "TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION",
    "TRADE_RL_ORACLE_COMPILE_MODE",
    "TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE",
)


def test_oracle_solver_environment_defaults_to_numpy(monkeypatch) -> None:
    for key in _ORACLE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    config = _oracle_solver_config()

    assert config.selection == "numpy"
    assert config.episode_batch_size == 8
    assert config.target_state_block_size is None
    assert config.cuda_memory_fraction == 0.65
    assert config.compile_mode == "disabled"
    assert config.compile_chunk_size == 16


def test_oracle_solver_environment_parses_explicit_cuda_contract(monkeypatch) -> None:
    monkeypatch.setenv("TRADE_RL_ORACLE_SOLVER", "cuda_or_numpy")
    monkeypatch.setenv("TRADE_RL_ORACLE_EPISODE_BATCH_SIZE", "4")
    monkeypatch.setenv("TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE", "32")
    monkeypatch.setenv("TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION", "0.5")
    monkeypatch.setenv("TRADE_RL_ORACLE_COMPILE_MODE", "reduce_overhead")
    monkeypatch.setenv("TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE", "8")

    config = _oracle_solver_config()

    assert config.selection == "cuda_or_numpy"
    assert config.episode_batch_size == 4
    assert config.target_state_block_size == 32
    assert config.cuda_memory_fraction == 0.5
    assert config.compile_mode == "reduce_overhead"
    assert config.compile_chunk_size == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TRADE_RL_ORACLE_SOLVER", "automatic"),
        ("TRADE_RL_ORACLE_EPISODE_BATCH_SIZE", "0"),
        ("TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE", "none"),
        ("TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION", "1.5"),
        ("TRADE_RL_ORACLE_COMPILE_MODE", "max-autotune"),
        ("TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE", "7"),
    ],
)
def test_oracle_solver_environment_rejects_invalid_values(
    monkeypatch, name: str, value: str
) -> None:
    for key in _ORACLE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        _oracle_solver_config()


def test_cuda_oracle_solver_requires_one_teacher_worker(monkeypatch) -> None:
    monkeypatch.setenv("TRADE_RL_TEACHER_WORKERS", "2")

    with pytest.raises(ValueError, match="TRADE_RL_TEACHER_WORKERS=1"):
        _teacher_worker_count(
            8,
            solver_config=sb3_training.OracleSolverConfig(selection="cuda"),
        )


def test_numpy_oracle_solver_defaults_to_one_compatibility_worker(monkeypatch) -> None:
    monkeypatch.delenv("TRADE_RL_TEACHER_WORKERS", raising=False)

    assert (
        _teacher_worker_count(8, solver_config=sb3_training.OracleSolverConfig()) == 1
    )


def test_oracle_episode_batch_cache_separates_solver_configs(monkeypatch) -> None:
    backend = object.__new__(StableBaselines3Backend)
    backend._oracle_episode_batch_cache = {}
    environment = SimpleNamespace(dataset=SimpleNamespace(dataset_id="f" * 64))
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    sampling = sb3_training.OracleEpisodeSamplingConfig(
        episode_bars=4,
        episode_count=2,
    )
    calls: list[str] = []

    def fake_build_episode_oracle_batch(*args, solver_config, **kwargs):
        del args, kwargs
        calls.append(solver_config.digest)
        return SimpleNamespace(solver_config=solver_config)

    monkeypatch.setattr(
        sb3_training,
        "build_episode_oracle_batch",
        fake_build_episode_oracle_batch,
    )
    numpy_result = backend._oracle_episode_batch(
        environment,
        (1, 10),
        teacher,
        sampling,
        solver_config=sb3_training.OracleSolverConfig(selection="numpy"),
    )
    cuda_result = backend._oracle_episode_batch(
        environment,
        (1, 10),
        teacher,
        sampling,
        solver_config=sb3_training.OracleSolverConfig(selection="cuda"),
    )

    assert numpy_result is not cuda_result
    assert len(calls) == 2
