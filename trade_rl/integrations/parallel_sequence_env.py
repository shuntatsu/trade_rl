"""Compact subprocess transport for structured sequence vector environments."""

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
        raise ValueError(
            "compact sequence observation already contains sequence fields"
        )
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
        if not callable(getattr(reconstructor, "reconstruct", None)):
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
