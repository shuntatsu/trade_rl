"""Training environment adapters for Stable-Baselines3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.rl.training import ResidualTrainingConfig

_HEAVY_TRAINING_INFO_KEYS = (
    "hybrid_execution",
    "shadow_execution",
    "hybrid_liquidation",
    "shadow_liquidation",
)


def _compact_training_info(info: dict[str, object]) -> dict[str, object]:
    """Keep callback diagnostics without copying the environment's histories.

    ``DummyVecEnv`` deep-copies every info mapping.  Execution results reference
    ``BookState`` objects whose return histories grow for the whole episode, so
    exposing them to SB3 turns rollout collection into quadratic work.  The raw
    Gymnasium environment retains its rich diagnostic contract; only the
    training adapter replaces those objects with the small fields consumed by
    telemetry and callbacks.
    """

    compact = dict(info)
    execution = compact.get("hybrid_execution")
    liquidation = compact.get("hybrid_liquidation")
    book = getattr(execution, "book", None)
    weights = getattr(book, "weights", None)
    if weights is not None:
        compact["telemetry_weights_after"] = np.asarray(
            weights,
            dtype=np.float64,
        ).copy()
    filled_turnover = getattr(execution, "filled_turnover", None)
    if filled_turnover is not None:
        compact["telemetry_filled_turnover"] = float(filled_turnover) + float(
            getattr(liquidation, "filled_turnover", 0.0)
        )
    fill_count = getattr(execution, "fill_count", None)
    if fill_count is not None:
        compact["telemetry_fill_count"] = int(fill_count) + int(
            getattr(liquidation, "fill_count", 0)
        )
    if "telemetry_risk_reasons" not in compact:
        risk = compact.get("hybrid_risk")
        reasons = getattr(risk, "reasons", ())
        compact["telemetry_risk_reasons"] = tuple(
            str(getattr(item, "value", item)) for item in reasons if str(item)
        )
    for key in _HEAVY_TRAINING_INFO_KEYS:
        compact.pop(key, None)
    return compact


class _TrainingInfoFilter(gym.Wrapper):
    """Remove history-bearing diagnostics before SB3 copies vector infos."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            _compact_training_info(info),
        )


def _filtered_training_environment(factory: Callable[[], Any]) -> Any:
    return _TrainingInfoFilter(factory())


class _FilteredEnvironmentFactory:
    """Preserve optional worker-index routing through the training info filter."""

    def __init__(self, factory: Callable[[], gym.Env[Any, Any]]) -> None:
        self.factory = factory

    def __call__(self) -> gym.Env[Any, Any]:
        return _filtered_training_environment(self.factory)

    def for_environment_index(self, index: int) -> Callable[[], gym.Env[Any, Any]]:
        indexed = getattr(self.factory, "for_environment_index", None)
        selected = indexed(index) if callable(indexed) else self.factory
        return partial(_filtered_training_environment, selected)


def _filtered_environment_factory(
    factory: Callable[[], gym.Env[Any, Any]],
) -> _FilteredEnvironmentFactory:
    return _FilteredEnvironmentFactory(factory)


def _environment_factories(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
) -> list[Callable[[], gym.Env[Any, Any]]]:
    indexed = getattr(factory, "for_environment_index", None)
    if callable(indexed):
        return [indexed(index) for index in range(n_envs)]
    return [factory for _ in range(n_envs)]


def _build_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = _environment_factories(factory, n_envs)
    if subprocesses:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def _effective_vector_environment_kind(config: ResidualTrainingConfig) -> str:
    if config.n_envs == 1:
        return "direct"
    if config.vector_environment_mode != "subprocess":
        return "in_process"
    if config.observation_encoder == "hierarchical_sequence_v2":
        return "subprocess_compact_sequence"
    return "subprocess"


def _compact_filtered_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
) -> gym.Env[Any, Any]:
    from trade_rl.rl.sequence_observations import (
        sequence_policy_plane_materialization,
    )

    with sequence_policy_plane_materialization(False):
        environment = factory()
    unwrapped: Any = getattr(environment, "unwrapped", environment)
    setter = getattr(unwrapped, "set_compact_sequence_training_observations", None)
    if not callable(setter):
        environment.close()
        raise TypeError(
            "parallel sequence worker does not support compact observations"
        )
    setter(True)
    return _TrainingInfoFilter(environment)


class _CompactFilteredEnvironmentFactory:
    """Preserve worker-index routing while enabling compact sequence observations."""

    def __init__(self, factory: Callable[[], gym.Env[Any, Any]]) -> None:
        self.factory = factory

    def __call__(self) -> gym.Env[Any, Any]:
        return _compact_filtered_training_environment(self.factory)

    def for_environment_index(self, index: int) -> Callable[[], gym.Env[Any, Any]]:
        indexed = getattr(self.factory, "for_environment_index", None)
        selected = indexed(index) if callable(indexed) else self.factory
        return partial(_compact_filtered_training_environment, selected)


def _compact_filtered_environment_factory(
    factory: Callable[[], gym.Env[Any, Any]],
) -> _CompactFilteredEnvironmentFactory:
    return _CompactFilteredEnvironmentFactory(factory)


def _build_parallel_sequence_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    full_observation_space: gym.spaces.Dict,
    reconstructor: Any,
) -> Any:
    from trade_rl.integrations.parallel_sequence_env import ParallelSequenceVecEnv

    workers = _build_training_environment(
        _compact_filtered_environment_factory(factory),
        n_envs,
        subprocesses=True,
    )
    try:
        return ParallelSequenceVecEnv(
            workers,
            full_observation_space=full_observation_space,
            reconstructor=reconstructor,
        )
    except BaseException:
        workers.close()
        raise


def _reset_observation_for_export(
    environment: object, *, seed: int
) -> Mapping[str, np.ndarray]:
    reset = getattr(environment, "reset", None)
    if not callable(reset):
        raise TypeError("structured export environment does not support reset")
    try:
        raw = reset(seed=seed)
    except TypeError:
        raw = reset()
    observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
    if not isinstance(observation, Mapping):
        raise ValueError("structured export requires a mapping observation")
    return {key: np.asarray(value) for key, value in observation.items()}
