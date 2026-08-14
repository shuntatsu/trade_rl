"""Learner-state dataset aggregation for behavior cloning.

This module is intentionally framework independent. The learner drives the
simulator, while the teacher labels exactly the states visited by that learner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.episode_teacher_artifact import EpisodeSupervisedPolicyDataset
from trade_rl.learning.teacher_artifact import ObservationBatch


class DaggerTeacher(Protocol):
    """Immutable causal teacher used to relabel learner-visited states."""

    @property
    def teacher_config_digest(self) -> str: ...

    def action_for(self, environment: Any, observation: object) -> np.ndarray: ...


def _identity(environment: Any, name: str) -> str:
    value = getattr(environment, name, None)
    if value is None and name == "dataset_id":
        value = getattr(getattr(environment, "dataset", None), "dataset_id", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"DAgger environment {name} must be a SHA-256 digest")
    return value


def _observation_copy(value: object) -> np.ndarray | dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("DAgger observation keys must be non-empty strings")
            array = np.asarray(value[key]).copy(order="C")
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError("DAgger observations must be finite numeric arrays")
            result[key] = array
        if not result:
            raise ValueError("DAgger structured observation must not be empty")
        return result
    array = np.asarray(value, dtype=np.float32).copy(order="C")
    if array.ndim == 0 or not np.isfinite(array).all():
        raise ValueError("DAgger observation must be finite and non-scalar")
    return array


def _stack_observations(
    values: list[np.ndarray | dict[str, np.ndarray]],
) -> ObservationBatch:
    if not values:
        raise ValueError("DAgger rollout contains no observations")
    first = values[0]
    if isinstance(first, dict):
        keys = tuple(first)
        if any(not isinstance(item, dict) or tuple(item) != keys for item in values):
            raise ValueError("DAgger structured observation schema drifted")
        return {
            key: np.stack(
                [np.asarray(item[key]) for item in values if isinstance(item, dict)],
                axis=0,
            )
            for key in keys
        }
    if any(isinstance(item, dict) for item in values):
        raise ValueError("DAgger observation transport changed within rollout")
    return np.stack([np.asarray(item) for item in values], axis=0)


def _observation_arrays(
    observations: ObservationBatch,
) -> tuple[tuple[str, np.ndarray], ...]:
    if isinstance(observations, Mapping):
        return tuple(
            (f"observation:{key}", np.asarray(observations[key]))
            for key in sorted(observations)
        )
    return (("observation", np.asarray(observations)),)


@dataclass(frozen=True, slots=True)
class DaggerEpisodeRollout:
    observations: ObservationBatch
    teacher_actions: np.ndarray
    learner_actions: np.ndarray
    decision_indices: np.ndarray
    dataset_id: str
    environment_digest: str
    action_spec_digest: str
    teacher_config_digest: str
    start: int
    stop: int
    initial_state_mode: str
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "dataset_id",
            "environment_digest",
            "action_spec_digest",
            "teacher_config_digest",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"DAgger {field} must be a SHA-256 digest")
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start + 1
        ):
            raise ValueError("DAgger episode range is invalid")
        if not self.initial_state_mode:
            raise ValueError("DAgger initial_state_mode must be non-empty")
        teacher = np.asarray(self.teacher_actions, dtype=np.float32).copy(order="C")
        learner = np.asarray(self.learner_actions, dtype=np.float32).copy(order="C")
        decisions = np.asarray(self.decision_indices, dtype=np.int64).reshape(-1).copy()
        if (
            teacher.ndim != 2
            or learner.shape != teacher.shape
            or len(teacher) == 0
            or decisions.shape != (len(teacher),)
            or not np.isfinite(teacher).all()
            or not np.isfinite(learner).all()
        ):
            raise ValueError("DAgger rollout arrays are not finite and sample aligned")
        if np.any(np.diff(decisions) <= 0):
            raise ValueError("DAgger decision indices must be strictly increasing")
        observation_count = (
            len(next(iter(self.observations.values())))
            if isinstance(self.observations, Mapping)
            else len(np.asarray(self.observations))
        )
        if observation_count != len(teacher):
            raise ValueError("DAgger observation count does not match actions")
        teacher.setflags(write=False)
        learner.setflags(write=False)
        decisions.setflags(write=False)
        object.__setattr__(self, "teacher_actions", teacher)
        object.__setattr__(self, "learner_actions", learner)
        object.__setattr__(self, "decision_indices", decisions)
        expected = content_and_arrays_digest(
            {
                "action_spec_digest": self.action_spec_digest,
                "dataset_id": self.dataset_id,
                "environment_digest": self.environment_digest,
                "initial_state_mode": self.initial_state_mode,
                "schema_version": "dagger_episode_rollout_v1",
                "start": self.start,
                "stop": self.stop,
                "teacher_config_digest": self.teacher_config_digest,
            },
            (
                *_observation_arrays(self.observations),
                ("teacher_actions", teacher),
                ("learner_actions", learner),
                ("decision_indices", decisions),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("DAgger rollout digest mismatch")
        object.__setattr__(self, "digest", expected)


def collect_dagger_episode(
    environment: Any,
    learner: Any,
    teacher: DaggerTeacher,
    *,
    start: int,
    stop: int,
    initial_state_mode: str,
) -> DaggerEpisodeRollout:
    """Collect teacher labels on states visited by the deterministic learner."""

    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start + 1
    ):
        raise ValueError("DAgger episode range is invalid")
    if not initial_state_mode:
        raise ValueError("DAgger initial_state_mode must be non-empty")
    predict = getattr(learner, "predict", None)
    label = getattr(teacher, "action_for", None)
    if not callable(predict):
        raise TypeError("DAgger learner must expose predict")
    if not callable(label):
        raise TypeError("DAgger teacher must expose action_for")
    teacher_digest = getattr(teacher, "teacher_config_digest", None)
    if not isinstance(teacher_digest, str) or len(teacher_digest) != 64:
        raise ValueError("DAgger teacher_config_digest must be a SHA-256 digest")

    decision_count = stop - start - 1
    observation, _ = environment.reset(
        options={
            "start_idx": start,
            "episode_bars": decision_count,
            "initial_state_mode": initial_state_mode,
        }
    )
    observations: list[np.ndarray | dict[str, np.ndarray]] = []
    teacher_actions: list[np.ndarray] = []
    learner_actions: list[np.ndarray] = []
    decisions: list[int] = []

    for offset in range(decision_count):
        current = int(getattr(environment, "current_index", start + offset))
        if current != start + offset:
            raise ValueError("DAgger environment advanced outside the requested range")
        observations.append(_observation_copy(observation))
        raw_teacher = np.asarray(
            label(environment, observation), dtype=np.float32
        ).reshape(-1)
        raw_learner, _ = predict(observation, deterministic=True)
        raw_learner = np.asarray(raw_learner, dtype=np.float32).reshape(-1)
        if raw_teacher.size == 0 or not np.isfinite(raw_teacher).all():
            raise ValueError("DAgger teacher action must be finite and non-empty")
        if raw_learner.shape != raw_teacher.shape or not np.isfinite(raw_learner).all():
            raise ValueError("DAgger learner action must match the finite teacher action")
        teacher_actions.append(raw_teacher.copy())
        learner_actions.append(raw_learner.copy())
        decisions.append(current)
        observation, _, terminated, truncated, _ = environment.step(raw_learner)
        if offset < decision_count - 1 and (terminated or truncated):
            raise ValueError("DAgger environment terminated before the requested stop")

    return DaggerEpisodeRollout(
        observations=_stack_observations(observations),
        teacher_actions=np.stack(teacher_actions, axis=0),
        learner_actions=np.stack(learner_actions, axis=0),
        decision_indices=np.asarray(decisions, dtype=np.int64),
        dataset_id=_identity(environment, "dataset_id"),
        environment_digest=_identity(environment, "environment_digest"),
        action_spec_digest=_identity(environment, "action_spec_digest"),
        teacher_config_digest=teacher_digest,
        start=start,
        stop=stop,
        initial_state_mode=initial_state_mode,
    )


def _concat_observations(
    base: ObservationBatch, additions: tuple[ObservationBatch, ...]
) -> ObservationBatch:
    if isinstance(base, Mapping):
        keys = tuple(sorted(base))
        if any(
            not isinstance(item, Mapping) or tuple(sorted(item)) != keys
            for item in additions
        ):
            raise ValueError("DAgger observation schema drifted while merging")
        return {
            key: np.concatenate(
                (np.asarray(base[key]), *(np.asarray(item[key]) for item in additions)),
                axis=0,
            )
            for key in keys
        }
    if any(isinstance(item, Mapping) for item in additions):
        raise ValueError("DAgger observation transport drifted while merging")
    return np.concatenate(
        (np.asarray(base), *(np.asarray(item) for item in additions)), axis=0
    )


def merge_dagger_rollouts(
    base: EpisodeSupervisedPolicyDataset,
    rollouts: tuple[DaggerEpisodeRollout, ...],
) -> EpisodeSupervisedPolicyDataset:
    """Append teacher labels from learner-state rollouts to one BC dataset."""

    if not isinstance(base, EpisodeSupervisedPolicyDataset):
        raise TypeError("DAgger base dataset must be episode aligned")
    items = tuple(rollouts)
    if not items:
        return base
    for rollout in items:
        if rollout.dataset_id != base.dataset_id:
            raise ValueError("DAgger rollout dataset identity drifted")
        if rollout.environment_digest != base.environment_digest:
            raise ValueError("DAgger rollout environment identity drifted")
        if rollout.action_spec_digest != base.action_spec_digest:
            raise ValueError("DAgger rollout action-spec identity drifted")
        if rollout.teacher_config_digest != base.teacher_config_digest:
            raise ValueError("DAgger rollout teacher identity drifted")
        if np.any(rollout.decision_indices < base.train_start) or np.any(
            rollout.decision_indices >= base.train_stop - 1
        ):
            raise ValueError("DAgger rollout leaves the base training envelope")
        if rollout.teacher_actions.shape[1] != base.actions.shape[1]:
            raise ValueError("DAgger rollout action width drifted")

    next_episode_id = base.episode_count
    appended_episode_ids = tuple(
        np.full(len(item.teacher_actions), next_episode_id + index, dtype=np.int64)
        for index, item in enumerate(items)
    )
    return EpisodeSupervisedPolicyDataset(
        observations=_concat_observations(
            base.observations,
            tuple(item.observations for item in items),
        ),
        actions=np.concatenate(
            (base.actions, *(item.teacher_actions for item in items)),
            axis=0,
        ),
        dataset_id=base.dataset_id,
        train_start=base.train_start,
        train_stop=base.train_stop,
        environment_digest=base.environment_digest,
        action_spec_digest=base.action_spec_digest,
        teacher_config_digest=base.teacher_config_digest,
        decision_indices=np.concatenate(
            (base.decision_indices, *(item.decision_indices for item in items)),
            axis=0,
        ),
        episode_ids=np.concatenate((base.episode_ids, *appended_episode_ids), axis=0),
        # Existing oracle provenance does not cover learner-state rows. Mixed
        # DAgger datasets deliberately drop solver provenance rather than
        # falsely extending it to newly aggregated samples.
        solver_provenance=None,
    )


__all__ = [
    "DaggerEpisodeRollout",
    "DaggerTeacher",
    "collect_dagger_episode",
    "merge_dagger_rollouts",
]
