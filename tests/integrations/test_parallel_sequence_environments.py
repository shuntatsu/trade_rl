from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from trade_rl.data.market import MarketDataset
from trade_rl.integrations import sb3_training
from trade_rl.integrations.parallel_sequence_env import (
    ParallelSequenceVecEnv,
    rehydrate_sequence_observations,
    rehydrate_terminal_observations,
)
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_observation import (
    EnvironmentObservationAssembler,
)
from trade_rl.rl.environment_observation_contract import (
    EnvironmentObservationContractBuilder,
)
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.sequence_observations import (
    sequence_policy_plane_materialization,
)
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 16,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 4,
        "n_envs": 2,
        "batch_size": 8,
        "n_epochs": 1,
        "observation_encoder": "flat_mlp",
        "device": "cpu",
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_vector_environment_mode_is_validated_and_identity_bound() -> None:
    automatic = _config()
    subprocess = _config(vector_environment_mode="subprocess")

    assert automatic.vector_environment_mode == "auto"
    assert subprocess.vector_environment_mode == "subprocess"
    assert automatic.digest_payload() != subprocess.digest_payload()

    with pytest.raises(ValueError, match="vector_environment_mode"):
        _config(vector_environment_mode="threads")


def test_maintained_full_configs_request_subprocess_vector_environments() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    direct = json.loads(
        (root / "examples/binance-multitimeframe/training-full.json").read_text(
            encoding="utf-8"
        )
    )["training"]
    walk_forward = json.loads(
        (root / "examples/binance-multitimeframe/walk-forward-full.json").read_text(
            encoding="utf-8"
        )
    )["candidates"][0]["run"]["training"]

    for training in (direct, walk_forward):
        assert training["vector_environment_mode"] == "subprocess"


class _Plane:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def components(self, index: int) -> dict[str, np.ndarray]:
        self.calls.append(index)
        return {
            "sequence_15m_values": np.asarray([[[index]]], dtype=np.float16),
            "sequence_15m_available": np.ones((1, 1, 1), dtype=np.uint8),
            "sequence_15m_staleness": np.zeros((1, 1, 1), dtype=np.float16),
        }


def test_compact_assembler_avoids_sequence_plane_and_matches_current_state() -> None:
    layout = ObservationLayout(
        n_symbols=2,
        n_features=1,
        action_size=1,
        n_factors=0,
        per_symbol_width=11,
        global_width=2,
    )
    plane = _Plane()
    assembler = EnvironmentObservationAssembler(
        SimpleNamespace(n_features=1),  # type: ignore[arg-type]
        observation_builder=object(),  # type: ignore[arg-type]
        layout=layout,
        normalizer=None,
        sequence_observation_builder=object(),  # type: ignore[arg-type]
        sequence_policy_plane=plane,  # type: ignore[arg-type]
        sequence_normalizer=None,
        action_size=1,
        n_factors=0,
        finite_horizon=False,
    )
    current = np.linspace(-0.5, 0.5, layout.size, dtype=np.float32)
    current[: layout.n_symbols * layout.per_symbol_width].reshape(
        layout.n_symbols, layout.per_symbol_width
    )[:, layout.current_weight_column] = np.asarray((0.25, -0.5))

    def flat_pair(
        self: object, *args: Any, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        del self, args, kwargs
        return current.copy(), current.copy()

    assembler.flat_pair = MethodType(flat_pair, assembler)  # type: ignore[method-assign]
    runtime = SimpleNamespace(current_index=9)
    arguments = {
        "trends": object(),
        "alpha": np.zeros(2),
        "factor_basis": np.empty((0, 2)),
        "pre_trade_risk": object(),
    }

    compact = assembler.compact_observation(runtime, **arguments)  # type: ignore[arg-type]

    assert plane.calls == []
    assert compact["decision_index"].tolist() == [9]
    assert not any(key.startswith("sequence_") for key in compact)

    full = assembler.observation(runtime, **arguments)  # type: ignore[arg-type]
    assert plane.calls == [9]
    for key in (
        "current_snapshot",
        "asset_state",
        "global_state",
        "active",
        "current_weights",
    ):
        np.testing.assert_array_equal(compact[key], full[key])


def _full_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
            "current_snapshot": spaces.Box(
                -np.inf, np.inf, shape=(1, 1), dtype=np.float32
            ),
            "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "sequence_15m_values": spaces.Box(
                -np.inf, np.inf, shape=(1, 1, 1), dtype=np.float16
            ),
            "sequence_15m_available": spaces.Box(0, 1, shape=(1, 1, 1), dtype=np.uint8),
            "sequence_15m_staleness": spaces.Box(
                0, np.inf, shape=(1, 1, 1), dtype=np.float16
            ),
        }
    )


def test_environment_compact_transport_switches_space_without_changing_full_contract() -> (
    None
):
    environment = object.__new__(ResidualMarketEnv)
    full_space = _full_space()
    environment._full_observation_space = full_space
    environment.observation_space = full_space
    environment.sequence_observation_builder = object()
    environment._compact_sequence_training_observations = False

    environment.set_compact_sequence_training_observations(True)
    assert tuple(environment.observation_space.spaces) == (
        "current_snapshot",
        "current_weights",
        "decision_index",
    )
    assert environment._compact_sequence_training_observations is True

    environment.set_compact_sequence_training_observations(False)
    assert environment.observation_space is full_space
    assert environment._compact_sequence_training_observations is False


def _market() -> MarketDataset:
    n_bars = 120
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n_bars) * np.timedelta64(
        15, "m"
    )
    close = np.linspace(100.0, 120.0, n_bars)[:, None]
    features = np.zeros((n_bars, 1, 4), dtype=np.float32)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC",),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack((close[0], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=("15m__ret", "1h__ret", "4h__ret", "1d__ret"),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _contract_builder(dataset: MarketDataset) -> EnvironmentObservationContractBuilder:
    return EnvironmentObservationContractBuilder(
        dataset,
        ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            structured_sequence_observation=True,
            sequence_windows=(("15m", 2), ("1h", 2), ("4h", 2), ("1d", 2)),
        ),
        action_spec=ActionSpec(),
        normalizer=None,
        sequence_normalizer=None,
        alpha_artifact_digest=None,
        factor_artifact_digest=None,
        action_spec_digest="b" * 64,
    )


def test_worker_construction_context_suppresses_sequence_policy_plane() -> None:
    dataset = _market()
    ordinary = _contract_builder(dataset).build(minimum_start_index=0)
    with sequence_policy_plane_materialization(False):
        compact_worker = _contract_builder(dataset).build(minimum_start_index=0)

    assert ordinary.sequence_policy_plane is not None
    assert compact_worker.sequence_policy_plane is None
    assert compact_worker.sequence_observation_builder is not None
    assert (
        compact_worker.observation_contract_digest
        == ordinary.observation_contract_digest
    )
    assert compact_worker.observation_space == ordinary.observation_space


class _Reconstructor:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def reconstruct(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        normalized = np.asarray(indices, dtype=np.int64).copy()
        self.calls.append(normalized)
        values = normalized.astype(np.float16).reshape(-1, 1, 1, 1)
        return {
            "sequence_15m_values": values,
            "sequence_15m_available": np.ones_like(values, dtype=np.uint8),
            "sequence_15m_staleness": np.zeros_like(values, dtype=np.float16),
        }


def _compact_batch(indices: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "decision_index": np.asarray(indices, dtype=np.int64).reshape(-1, 1),
        "current_snapshot": np.asarray(indices, dtype=np.float32).reshape(-1, 1, 1),
        "current_weights": np.zeros((len(indices), 1), dtype=np.float32),
    }


def test_parent_rehydration_batches_all_current_indices_once() -> None:
    reconstructor = _Reconstructor()
    compact = _compact_batch((7, 11, 13))

    full = rehydrate_sequence_observations(compact, reconstructor)  # type: ignore[arg-type]

    assert len(reconstructor.calls) == 1
    np.testing.assert_array_equal(reconstructor.calls[0], (7, 11, 13))
    np.testing.assert_array_equal(full["current_snapshot"], compact["current_snapshot"])
    assert full["sequence_15m_values"].shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(
        full["sequence_15m_values"].reshape(-1),
        np.asarray((7, 11, 13), dtype=np.float16),
    )


def test_terminal_rehydration_batches_terminal_indices_once() -> None:
    reconstructor = _Reconstructor()
    infos: list[dict[str, object]] = [
        {"terminal_observation": _compact_batch((5,)) | {"tag": np.asarray([1])}},
        {"other": True},
        {"terminal_observation": _compact_batch((9,)) | {"tag": np.asarray([2])}},
    ]

    resolved = rehydrate_terminal_observations(infos, reconstructor)  # type: ignore[arg-type]

    assert len(reconstructor.calls) == 1
    np.testing.assert_array_equal(reconstructor.calls[0], (5, 9))
    first = resolved[0]["terminal_observation"]
    third = resolved[2]["terminal_observation"]
    assert isinstance(first, dict)
    assert isinstance(third, dict)
    assert first["sequence_15m_values"].shape == (1, 1, 1)
    assert third["sequence_15m_values"].shape == (1, 1, 1)
    assert float(first["sequence_15m_values"].reshape(-1)[0]) == 5.0
    assert float(third["sequence_15m_values"].reshape(-1)[0]) == 9.0
    assert resolved[1] == {"other": True}


class _FakeVecEnv(VecEnv):
    def __init__(self) -> None:
        compact_space = spaces.Dict(
            {
                "decision_index": spaces.Box(0, 100, shape=(1,), dtype=np.int64),
                "current_snapshot": spaces.Box(
                    -np.inf, np.inf, shape=(1, 1), dtype=np.float32
                ),
            }
        )
        super().__init__(
            2,
            compact_space,
            spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        )
        self.pending_actions: np.ndarray | None = None

    def reset(self) -> dict[str, np.ndarray]:
        return _compact_batch((3, 4))

    def step_async(self, actions: np.ndarray) -> None:
        self.pending_actions = actions

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        return (
            _compact_batch((6, 8)),
            np.asarray((0.1, 0.2), dtype=np.float32),
            np.asarray((True, False), dtype=np.bool_),
            [
                {"terminal_observation": _compact_batch((5,))},
                {},
            ],
        )

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        del indices
        if attr_name == "render_mode":
            return [None, None]
        return []

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        del attr_name, value, indices

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        del method_name, method_args, indices, method_kwargs
        return []

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: Any = None
    ) -> list[bool]:
        del wrapper_class, indices
        return [False, False]


def test_parallel_wrapper_rehydrates_reset_step_and_terminal_observations() -> None:
    reconstructor = _Reconstructor()
    wrapper = ParallelSequenceVecEnv(
        _FakeVecEnv(),
        full_observation_space=_full_space(),
        reconstructor=reconstructor,  # type: ignore[arg-type]
    )

    reset = wrapper.reset()
    wrapper.step_async(np.zeros((2, 1), dtype=np.float32))
    step, rewards, dones, infos = wrapper.step_wait()

    assert wrapper.observation_space == _full_space()
    assert reset["sequence_15m_values"].shape == (2, 1, 1, 1)
    assert step["sequence_15m_values"].shape == (2, 1, 1, 1)
    assert rewards.tolist() == pytest.approx((0.1, 0.2))
    assert dones.tolist() == [True, False]
    terminal = infos[0]["terminal_observation"]
    assert terminal["sequence_15m_values"].shape == (1, 1, 1)
    assert len(reconstructor.calls) == 3


class _WorkerEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.compact_enabled = False

    @property
    def unwrapped(self) -> _WorkerEnvironment:
        return self

    def set_compact_sequence_training_observations(self, enabled: bool) -> None:
        self.compact_enabled = enabled

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_parallel_sequence_builder_uses_spawn_workers_and_parent_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_environment = _WorkerEnvironment()
    workers = object()
    wrapped = object()
    observed: dict[str, object] = {}

    def build(
        worker_factory: Any,
        n_envs: int,
        *,
        subprocesses: bool = True,
    ) -> object:
        observed["n_envs"] = n_envs
        observed["subprocesses"] = subprocesses
        worker = worker_factory()
        observed["worker_compact"] = worker.unwrapped.compact_enabled
        return workers

    class FakeWrapper:
        def __new__(
            cls,
            environment: object,
            *,
            full_observation_space: spaces.Dict,
            reconstructor: object,
        ) -> object:
            del cls
            observed["workers"] = environment
            observed["full_space"] = full_observation_space
            observed["reconstructor"] = reconstructor
            return wrapped

    monkeypatch.setattr(sb3_training, "_build_training_environment", build)
    monkeypatch.setattr(
        "trade_rl.integrations.parallel_sequence_env.ParallelSequenceVecEnv",
        FakeWrapper,
    )
    reconstructor = object()
    full_space = _full_space()

    result = sb3_training._build_parallel_sequence_training_environment(
        lambda: worker_environment,
        4,
        full_observation_space=full_space,
        reconstructor=reconstructor,
    )

    assert result is wrapped
    assert observed == {
        "n_envs": 4,
        "subprocesses": True,
        "worker_compact": True,
        "workers": workers,
        "full_space": full_space,
        "reconstructor": reconstructor,
    }


@pytest.mark.parametrize(
    ("n_envs", "sequence", "mode", "expected"),
    (
        (1, True, "subprocess", "direct"),
        (4, True, "auto", "in_process"),
        (4, True, "in_process", "in_process"),
        (4, True, "subprocess", "subprocess_compact_sequence"),
        (4, False, "auto", "in_process"),
        (4, False, "in_process", "in_process"),
        (4, False, "subprocess", "subprocess"),
    ),
)
def test_effective_vector_environment_kind_is_explicit(
    n_envs: int,
    sequence: bool,
    mode: str,
    expected: str,
) -> None:
    config = _config(
        n_envs=n_envs,
        n_steps=8 if n_envs == 1 else 4,
        batch_size=8,
        observation_encoder=("hierarchical_sequence_v2" if sequence else "flat_mlp"),
        policy="MultiInputPolicy" if sequence else "MlpPolicy",
        vector_environment_mode=mode,
    )

    assert sb3_training._effective_vector_environment_kind(config) == expected
