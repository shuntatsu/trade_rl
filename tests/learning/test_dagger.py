from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.learning.dagger import (
    DaggerEpisodeRollout,
    collect_dagger_episode,
    merge_dagger_rollouts,
)
from trade_rl.learning.episode_teacher_artifact import EpisodeSupervisedPolicyDataset

_DATASET = "1" * 64
_ENVIRONMENT = "2" * 64
_ACTION_SPEC = "3" * 64
_TEACHER = "4" * 64


class _Learner:
    def __init__(self, action: float) -> None:
        self._action = float(action)

    def predict(self, observation, *, deterministic: bool):
        assert deterministic is True
        return np.asarray([self._action], dtype=np.float32), None


class _VectorLearner:
    def __init__(self, action: tuple[float, ...]) -> None:
        self._action = np.asarray(action, dtype=np.float32)

    def predict(self, observation, *, deterministic: bool):
        assert deterministic is True
        return self._action.copy(), None


@dataclass(frozen=True)
class _Teacher:
    teacher_config_digest: str = _TEACHER
    action: float = -0.75

    def action_for(self, environment, observation) -> np.ndarray:
        del environment, observation
        return np.asarray([self.action], dtype=np.float32)


class _Environment:
    dataset_id = _DATASET
    environment_digest = _ENVIRONMENT
    action_spec_digest = _ACTION_SPEC

    def __init__(self) -> None:
        self.current_index = -1
        self.stop_index = -1
        self.received_actions: list[np.ndarray] = []

    def _observation(self) -> np.ndarray:
        return np.asarray([float(self.current_index), 1.0], dtype=np.float32)

    def reset(self, *, options: dict[str, object]):
        self.current_index = int(options["start_idx"])
        self.stop_index = self.current_index + int(options["episode_bars"])
        return self._observation(), {}

    def step(self, action: np.ndarray):
        resolved = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        self.received_actions.append(resolved)
        self.current_index += 1
        terminated = self.current_index >= self.stop_index
        return self._observation(), 0.0, terminated, False, {}


def _base_dataset() -> EpisodeSupervisedPolicyDataset:
    return EpisodeSupervisedPolicyDataset(
        observations=np.asarray([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32),
        actions=np.asarray([[0.1], [0.2]], dtype=np.float32),
        dataset_id=_DATASET,
        train_start=1,
        train_stop=6,
        environment_digest=_ENVIRONMENT,
        action_spec_digest=_ACTION_SPEC,
        teacher_config_digest=_TEACHER,
        decision_indices=np.asarray([1, 2], dtype=np.int64),
        episode_ids=np.asarray([0, 0], dtype=np.int64),
    )


def test_dagger_labels_learner_visited_states_but_steps_only_learner_actions() -> None:
    environment = _Environment()

    rollout = collect_dagger_episode(
        environment,
        _Learner(0.5),
        _Teacher(),
        start=3,
        stop=6,
        initial_state_mode="cash",
    )

    assert rollout.decision_indices.tolist() == [3, 4]
    assert rollout.teacher_actions[:, 0].tolist() == pytest.approx([-0.75, -0.75])
    assert rollout.learner_actions[:, 0].tolist() == pytest.approx([0.5, 0.5])
    assert [item.item() for item in environment.received_actions] == pytest.approx(
        [0.5, 0.5]
    )
    assert np.asarray(rollout.observations)[:, 0].tolist() == pytest.approx([3.0, 4.0])
    assert rollout.teacher_config_digest == _TEACHER
    assert len(rollout.digest) == 64


def test_dagger_rejects_non_finite_teacher_labels() -> None:
    with pytest.raises(ValueError, match="teacher action"):
        collect_dagger_episode(
            _Environment(),
            _Learner(0.5),
            _Teacher(action=float("nan")),
            start=3,
            stop=6,
            initial_state_mode="cash",
        )


def test_dagger_rejects_non_finite_or_dimension_mismatched_learner_action() -> None:
    with pytest.raises(ValueError, match="learner action"):
        collect_dagger_episode(
            _Environment(),
            _Learner(float("nan")),
            _Teacher(),
            start=3,
            stop=6,
            initial_state_mode="cash",
        )
    with pytest.raises(ValueError, match="learner action"):
        collect_dagger_episode(
            _Environment(),
            _VectorLearner((0.5, 0.25)),
            _Teacher(),
            start=3,
            stop=6,
            initial_state_mode="cash",
        )


def test_dagger_rollout_copies_and_freezes_observations() -> None:
    observations = np.asarray([[3.0, 1.0], [4.0, 1.0]], dtype=np.float32)
    rollout = DaggerEpisodeRollout(
        observations=observations,
        teacher_actions=np.asarray([[-0.75], [-0.75]], dtype=np.float32),
        learner_actions=np.asarray([[0.5], [0.5]], dtype=np.float32),
        decision_indices=np.asarray([3, 4], dtype=np.int64),
        dataset_id=_DATASET,
        environment_digest=_ENVIRONMENT,
        action_spec_digest=_ACTION_SPEC,
        teacher_config_digest=_TEACHER,
        start=3,
        stop=6,
        initial_state_mode="cash",
    )

    observations[0, 0] = 99.0

    frozen = np.asarray(rollout.observations)
    assert frozen[0, 0] == pytest.approx(3.0)
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0, 0] = 42.0


def test_dagger_rollout_rejects_structured_observation_count_drift() -> None:
    with pytest.raises(ValueError, match="observation count"):
        DaggerEpisodeRollout(
            observations={
                "asset": np.asarray([[3.0], [4.0]], dtype=np.float32),
                "global": np.asarray([[1.0]], dtype=np.float32),
            },
            teacher_actions=np.asarray([[-0.75], [-0.75]], dtype=np.float32),
            learner_actions=np.asarray([[0.5], [0.5]], dtype=np.float32),
            decision_indices=np.asarray([3, 4], dtype=np.int64),
            dataset_id=_DATASET,
            environment_digest=_ENVIRONMENT,
            action_spec_digest=_ACTION_SPEC,
            teacher_config_digest=_TEACHER,
            start=3,
            stop=6,
            initial_state_mode="cash",
        )


def test_merge_dagger_rollout_appends_new_contiguous_episode_ids() -> None:
    base = _base_dataset()
    rollout = collect_dagger_episode(
        _Environment(),
        _Learner(0.5),
        _Teacher(),
        start=3,
        stop=6,
        initial_state_mode="cash",
    )

    merged = merge_dagger_rollouts(base, (rollout,))

    assert merged.actions[:, 0].tolist() == pytest.approx([0.1, 0.2, -0.75, -0.75])
    assert merged.decision_indices.tolist() == [1, 2, 3, 4]
    assert merged.episode_ids.tolist() == [0, 0, 1, 1]
    assert merged.episode_count == 2
    assert merged.dataset_id == base.dataset_id
    assert merged.environment_digest == base.environment_digest
    assert merged.action_spec_digest == base.action_spec_digest
    assert merged.teacher_config_digest == base.teacher_config_digest


def test_merge_dagger_rollouts_rejects_dataset_and_teacher_identity_drift() -> None:
    base = _base_dataset()
    environment = _Environment()
    environment.dataset_id = "5" * 64
    dataset_drift = collect_dagger_episode(
        environment,
        _Learner(0.5),
        _Teacher(),
        start=3,
        stop=6,
        initial_state_mode="cash",
    )

    with pytest.raises(ValueError, match="dataset identity"):
        merge_dagger_rollouts(base, (dataset_drift,))

    teacher_drift = collect_dagger_episode(
        _Environment(),
        _Learner(0.5),
        _Teacher(teacher_config_digest="6" * 64),
        start=3,
        stop=6,
        initial_state_mode="cash",
    )
    with pytest.raises(ValueError, match="teacher identity"):
        merge_dagger_rollouts(base, (teacher_drift,))
