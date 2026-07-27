from __future__ import annotations

import json
from pathlib import Path


def replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{seam} changed: expected one exact match, got {count}")
    return source.replace(old, new)


parallel_path = Path("trade_rl/integrations/parallel_sequence_env.py")
parallel_path.write_text(
    '''"""Compact subprocess transport for structured sequence vector environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from trade_rl.integrations.compact_rollout_buffer import SequenceRolloutReconstructor

_SEQUENCE_PREFIX = "sequence_"
_DECISION_INDEX_KEY = "decision_index"


def rehydrate_sequence_observations(
    observations: Mapping[str, np.ndarray],
    reconstructor: SequenceRolloutReconstructor,
) -> dict[str, np.ndarray]:
    """Restore one vectorized compact observation with a single batch gather."""

    if _DECISION_INDEX_KEY not in observations:
        raise ValueError("compact sequence observation is missing decision_index")
    if any(key.startswith(_SEQUENCE_PREFIX) for key in observations):
        raise ValueError("compact sequence observation already contains sequence fields")
    raw_indices = np.asarray(observations[_DECISION_INDEX_KEY])
    if raw_indices.ndim != 2 or raw_indices.shape[1] != 1:
        raise ValueError("vector decision_index must have shape [n_envs, 1]")
    indices = np.asarray(raw_indices[:, 0], dtype=np.int64)
    components = reconstructor.reconstruct(indices)
    result = {key: np.asarray(value) for key, value in observations.items()}
    for key, value in components.items():
        component = np.asarray(value)
        if not key.startswith(_SEQUENCE_PREFIX):
            raise ValueError("sequence reconstructor returned a non-sequence field")
        if component.shape[0] != indices.size:
            raise ValueError("sequence reconstruction batch size mismatch")
        result[key] = component
    return result


def rehydrate_terminal_observations(
    infos: Sequence[Mapping[str, Any]],
    reconstructor: SequenceRolloutReconstructor,
) -> list[dict[str, Any]]:
    """Restore all compact terminal observations with one reconstruction call."""

    resolved = [dict(info) for info in infos]
    records: list[tuple[int, dict[str, np.ndarray], int]] = []
    for info_index, info in enumerate(resolved):
        terminal = info.get("terminal_observation")
        if terminal is None:
            continue
        if not isinstance(terminal, Mapping):
            raise TypeError("terminal_observation must be a structured mapping")
        compact = {key: np.asarray(value) for key, value in terminal.items()}
        if _DECISION_INDEX_KEY not in compact:
            raise ValueError("compact terminal observation is missing decision_index")
        decision = np.asarray(compact[_DECISION_INDEX_KEY]).reshape(-1)
        if decision.shape != (1,):
            raise ValueError("terminal decision_index must contain one value")
        records.append((info_index, compact, int(decision[0])))
    if not records:
        return resolved

    indices = np.asarray([record[2] for record in records], dtype=np.int64)
    components = reconstructor.reconstruct(indices)
    for row, (info_index, compact, _decision) in enumerate(records):
        hydrated = dict(compact)
        for key, value in components.items():
            component = np.asarray(value)
            if component.shape[0] != len(records):
                raise ValueError("terminal sequence reconstruction batch size mismatch")
            hydrated[key] = component[row]
        resolved[info_index]["terminal_observation"] = hydrated
    return resolved


class ParallelSequenceVecEnv(VecEnvWrapper):
    """Expose full sequence observations while workers exchange compact state."""

    def __init__(
        self,
        venv: VecEnv,
        *,
        full_observation_space: spaces.Dict,
        reconstructor: SequenceRolloutReconstructor,
    ) -> None:
        if not isinstance(full_observation_space, spaces.Dict):
            raise TypeError("parallel sequence environment requires a Dict space")
        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("parallel sequence environment reconstructor is invalid")
        self.reconstructor = reconstructor
        super().__init__(venv, observation_space=full_observation_space)

    def reset(self) -> dict[str, np.ndarray]:
        observations = self.venv.reset()
        if not isinstance(observations, Mapping):
            raise TypeError("compact vector reset must return a mapping")
        return rehydrate_sequence_observations(observations, self.reconstructor)

    def step_wait(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        observations, rewards, dones, infos = self.venv.step_wait()
        if not isinstance(observations, Mapping):
            raise TypeError("compact vector step must return a mapping")
        full_observations = rehydrate_sequence_observations(
            observations,
            self.reconstructor,
        )
        full_infos = rehydrate_terminal_observations(infos, self.reconstructor)
        return full_observations, rewards, dones, full_infos


__all__ = [
    "ParallelSequenceVecEnv",
    "rehydrate_sequence_observations",
    "rehydrate_terminal_observations",
]
''',
    encoding="utf-8",
)


training_path = Path("trade_rl/rl/training.py")
training = training_path.read_text(encoding="utf-8")
training = replace_once(
    training,
    '''    checkpoint_interval_steps: int | None = None
    max_checkpoints: int = 5
    n_envs: int = 1
    behavior_cloning_epochs: int = 0
''',
    '''    checkpoint_interval_steps: int | None = None
    max_checkpoints: int = 5
    n_envs: int = 1
    vector_environment_mode: str = "auto"
    behavior_cloning_epochs: int = 0
''',
    seam="vector environment field",
)
training = replace_once(
    training,
    '''        if not isinstance(self.sequence_encoder, bool):
            raise ValueError("sequence_encoder must be a boolean")
''',
    '''        if self.vector_environment_mode not in {
            "auto",
            "in_process",
            "subprocess",
        }:
            raise ValueError(
                "vector_environment_mode must be auto, in_process, or subprocess"
            )
        if not isinstance(self.sequence_encoder, bool):
            raise ValueError("sequence_encoder must be a boolean")
''',
    seam="vector environment validation",
)
training = replace_once(
    training,
    '''            "n_envs": self.n_envs,
            "n_steps": self.n_steps,
''',
    '''            "n_envs": self.n_envs,
            "vector_environment_mode": self.vector_environment_mode,
            "n_steps": self.n_steps,
''',
    seam="vector environment digest",
)
training_path.write_text(training, encoding="utf-8")


sequence_path = Path("trade_rl/rl/sequence_observations.py")
sequence = sequence_path.read_text(encoding="utf-8")
sequence = replace_once(
    sequence,
    '''import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Protocol
''',
    '''import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Protocol
''',
    seam="sequence materialization imports",
)
sequence = replace_once(
    sequence,
    '''_FLOAT16_MAX = float(np.finfo(np.float16).max)


class SequenceNormalizerProtocol''',
    '''_FLOAT16_MAX = float(np.finfo(np.float16).max)
_SEQUENCE_POLICY_PLANE_MATERIALIZATION: ContextVar[bool] = ContextVar(
    "sequence_policy_plane_materialization",
    default=True,
)


@contextmanager
def sequence_policy_plane_materialization(enabled: bool) -> Iterator[None]:
    """Temporarily control policy-plane construction for worker environments."""

    if not isinstance(enabled, bool):
        raise TypeError("sequence policy-plane materialization flag must be boolean")
    token = _SEQUENCE_POLICY_PLANE_MATERIALIZATION.set(enabled)
    try:
        yield
    finally:
        _SEQUENCE_POLICY_PLANE_MATERIALIZATION.reset(token)


def should_materialize_sequence_policy_plane() -> bool:
    return bool(_SEQUENCE_POLICY_PLANE_MATERIALIZATION.get())


class SequenceNormalizerProtocol''',
    seam="sequence materialization context",
)
sequence = replace_once(
    sequence,
    '''    "build_sequence_policy_plane",
    "build_structured_current_observation",
''',
    '''    "build_sequence_policy_plane",
    "build_structured_current_observation",
    "sequence_policy_plane_materialization",
    "should_materialize_sequence_policy_plane",
''',
    seam="sequence context exports",
)
sequence_path.write_text(sequence, encoding="utf-8")


contract_path = Path("trade_rl/rl/environment_observation_contract.py")
contract = contract_path.read_text(encoding="utf-8")
contract = replace_once(
    contract,
    '''    SequenceWindowSpec,
    build_sequence_policy_plane,
)
''',
    '''    SequenceWindowSpec,
    build_sequence_policy_plane,
    should_materialize_sequence_policy_plane,
)
''',
    seam="contract materialization import",
)
contract = replace_once(
    contract,
    '''            sequence_policy_plane = build_sequence_policy_plane(
                self.dataset,
                sequence_observation_builder,
                self.sequence_normalizer,
            )
''',
    '''            if should_materialize_sequence_policy_plane():
                sequence_policy_plane = build_sequence_policy_plane(
                    self.dataset,
                    sequence_observation_builder,
                    self.sequence_normalizer,
                )
''',
    seam="conditional sequence policy plane",
)
contract_path.write_text(contract, encoding="utf-8")


observation_path = Path("trade_rl/rl/environment_observation.py")
observation = observation_path.read_text(encoding="utf-8")
observation = replace_once(
    observation,
    '''    def observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        if self.sequence_observation_builder is None:
            return current
        if self.sequence_policy_plane is not None:
            structured = build_structured_current_observation(
                current_flat=current,
                layout=self.layout,
                n_features=self.dataset.n_features,
            )
            structured.update(
                self.sequence_policy_plane.components(runtime.current_index)
            )
            structured["decision_index"] = np.asarray(
                [runtime.current_index],
                dtype=np.int64,
            )
            return structured
        sequence = self.sequence_observation_builder.build(
            self.dataset,
            index=runtime.current_index,
        )
        structured = build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured
''',
    '''    def compact_observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> dict[str, np.ndarray]:
        """Build current structured state without sequence policy channels."""

        if self.sequence_observation_builder is None:
            raise RuntimeError("compact observation requires a sequence contract")
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        structured = build_structured_current_observation(
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
        )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured

    def observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.sequence_observation_builder is None:
            _, current = self.flat_pair(
                runtime,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                pre_trade_risk=pre_trade_risk,
            )
            return current
        structured = self.compact_observation(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        if self.sequence_policy_plane is not None:
            structured.update(
                self.sequence_policy_plane.components(runtime.current_index)
            )
            return structured
        sequence = self.sequence_observation_builder.build(
            self.dataset,
            index=runtime.current_index,
        )
        sequence_components = build_structured_policy_observation(
            sequence=sequence,
            current_flat=np.concatenate(
                (
                    structured["current_snapshot"].reshape(-1),
                    structured["asset_state"].reshape(-1),
                    structured["global_state"].reshape(-1),
                )
            ),
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        for key, value in sequence_components.items():
            if key.startswith("sequence_"):
                structured[key] = value
        return structured
''',
    seam="compact observation assembly",
)
observation_path.write_text(observation, encoding="utf-8")


environment_path = Path("trade_rl/rl/environment.py")
environment = environment_path.read_text(encoding="utf-8")
environment = replace_once(
    environment,
    '''        self.observation_space = observation_contract.observation_space
        self.action_space = observation_contract.action_space
''',
    '''        self.observation_space = observation_contract.observation_space
        self._full_observation_space = observation_contract.observation_space
        self._compact_sequence_training_observations = False
        self.action_space = observation_contract.action_space
''',
    seam="environment full observation space",
)
environment = replace_once(
    environment,
    '''    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.observation(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )
''',
    '''    def set_compact_sequence_training_observations(self, enabled: bool) -> None:
        """Switch only the runtime transport, never the observation identity."""

        if not isinstance(enabled, bool):
            raise TypeError("compact sequence observation flag must be boolean")
        if enabled and self.sequence_observation_builder is None:
            raise RuntimeError("compact sequence observations require a sequence contract")
        self._compact_sequence_training_observations = enabled
        if not enabled:
            self.observation_space = self._full_observation_space
            return
        if not isinstance(self._full_observation_space, gym.spaces.Dict):
            raise RuntimeError("compact sequence observations require a Dict space")
        self.observation_space = gym.spaces.Dict(
            {
                key: value
                for key, value in self._full_observation_space.spaces.items()
                if not key.startswith("sequence_")
            }
        )

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        builder = (
            self._observation_assembler.compact_observation
            if self._compact_sequence_training_observations
            else self._observation_assembler.observation
        )
        return builder(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )
''',
    seam="environment compact transport",
)
environment_path.write_text(environment, encoding="utf-8")


sb3_path = Path("trade_rl/integrations/sb3_training.py")
sb3 = sb3_path.read_text(encoding="utf-8")
sb3 = replace_once(
    sb3,
    '''import tempfile
from collections.abc import Callable, Mapping
''',
    '''import tempfile
from collections.abc import Callable, Mapping
from functools import partial
''',
    seam="parallel environment partial import",
)
sb3 = replace_once(
    sb3,
    '''def _build_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [factory for _ in range(n_envs)]
    if subprocesses:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


''',
    '''def _build_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [factory for _ in range(n_envs)]
    if subprocesses:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def _effective_vector_environment_kind(config: ResidualTrainingConfig) -> str:
    if config.n_envs == 1:
        return "direct"
    if config.vector_environment_mode == "in_process":
        return "in_process"
    if config.sequence_encoder:
        return (
            "subprocess_compact_sequence"
            if config.vector_environment_mode == "subprocess"
            else "in_process"
        )
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


def _build_parallel_sequence_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    full_observation_space: gym.spaces.Dict,
    reconstructor: Any,
) -> Any:
    from trade_rl.integrations.parallel_sequence_env import ParallelSequenceVecEnv

    workers = _build_training_environment(
        partial(_compact_filtered_training_environment, factory),
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


''',
    seam="parallel sequence environment helpers",
)
sb3 = replace_once(
    sb3,
    '''            if config.n_envs == 1:
                environment = _TrainingInfoFilter(probe)
                probe = None
            else:
                probe_to_close = probe
                probe = None
                probe_to_close.close()
                environment = _build_training_environment(
                    lambda: _filtered_training_environment(self.environment_factory),
                    config.n_envs,
                    subprocesses=False,
                )
''',
    '''            vector_environment_kind = _effective_vector_environment_kind(config)
            full_observation_space = probe.observation_space
            if config.n_envs == 1:
                environment = _TrainingInfoFilter(probe)
                probe = None
            else:
                probe_to_close = probe
                probe = None
                probe_to_close.close()
                if vector_environment_kind == "subprocess_compact_sequence":
                    if sequence_reconstructor is None:
                        raise RuntimeError(
                            "parallel sequence environment requires a reconstructor"
                        )
                    if not isinstance(full_observation_space, gym.spaces.Dict):
                        raise RuntimeError(
                            "parallel sequence environment requires a Dict space"
                        )
                    environment = _build_parallel_sequence_training_environment(
                        self.environment_factory,
                        config.n_envs,
                        full_observation_space=full_observation_space,
                        reconstructor=sequence_reconstructor,
                    )
                else:
                    environment = _build_training_environment(
                        lambda: _filtered_training_environment(
                            self.environment_factory
                        ),
                        config.n_envs,
                        subprocesses=vector_environment_kind == "subprocess",
                    )
''',
    seam="backend vector environment selection",
)
sb3 = replace_once(
    sb3,
    '''                        "vector_environment": (
                            "native" if config.n_envs == 1 else "dummy"
                        ),
''',
    '''                        "vector_environment": vector_environment_kind,
''',
    seam="vector environment architecture evidence",
)
sb3_path.write_text(sb3, encoding="utf-8")


for config_path in (
    Path("examples/binance-multitimeframe/training-full.json"),
    Path("examples/binance-multitimeframe/walk-forward-full.json"),
):
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    training_payload = (
        payload["training"]
        if config_path.name == "training-full.json"
        else payload["candidates"][0]["run"]["training"]
    )
    training_payload["vector_environment_mode"] = "subprocess"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
