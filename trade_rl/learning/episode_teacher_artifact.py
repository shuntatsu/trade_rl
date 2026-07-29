"""Immutable episode-aligned supervised datasets for Oracle behavior cloning."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.learning.teacher_artifact import (
    TEACHER_ARRAYS_NAME,
    TEACHER_MANIFEST_NAME,
    ObservationBatch,
    ObservationValue,
    SupervisedPolicyDataset,
    TeacherRolloutEnvironment,
    _array_digest,
    _atomic_write,
    _deterministic_npz,
    _normalize_observations,
    _observation_arrays,
    _observation_digest,
    _sha256,
)

EPISODE_TEACHER_ARTIFACT_SCHEMA: Final = "episode_supervised_teacher_artifact_v1"
_ALLOWED_FILES = frozenset({TEACHER_MANIFEST_NAME, TEACHER_ARRAYS_NAME})
_COMPACT_KEYS = (
    "active",
    "asset_state",
    "current_snapshot",
    "current_weights",
    "global_state",
)


def _readonly_int_vector(value: object, *, field: str, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != count or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{field} must be a sample-aligned integer vector")
    resolved = np.asarray(raw, dtype=np.int64).copy(order="C")
    resolved.setflags(write=False)
    return resolved


@dataclass(frozen=True, slots=True)
class EpisodeSupervisedPolicyDataset(SupervisedPolicyDataset):
    """Supervised actions with explicit non-contiguous episode provenance."""

    decision_indices: np.ndarray
    episode_ids: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("action_spec_digest", self.action_spec_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            require_sha256(value, field=name)
        if (
            isinstance(self.train_start, bool)
            or isinstance(self.train_stop, bool)
            or not isinstance(self.train_start, int)
            or not isinstance(self.train_stop, int)
            or self.train_start < 0
            or self.train_stop <= self.train_start + 1
        ):
            raise ValueError("episode teacher training envelope is invalid")
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError("episode teacher actions must be a non-empty finite matrix")
        observations = _normalize_observations(
            self.observations,
            expected_count=len(actions),
        )
        decision_indices = _readonly_int_vector(
            self.decision_indices,
            field="decision_indices",
            count=len(actions),
        )
        episode_ids = _readonly_int_vector(
            self.episode_ids,
            field="episode_ids",
            count=len(actions),
        )
        if np.any(decision_indices < self.train_start) or np.any(
            decision_indices >= self.train_stop - 1
        ):
            raise ValueError("episode teacher decision indices leave the train envelope")
        if np.any(episode_ids < 0):
            raise ValueError("episode teacher ids must be non-negative")
        unique_ids = np.unique(episode_ids)
        if not np.array_equal(unique_ids, np.arange(len(unique_ids), dtype=np.int64)):
            raise ValueError("episode teacher ids must be contiguous from zero")
        for episode_id in unique_ids:
            mask = episode_ids == episode_id
            positions = np.flatnonzero(mask)
            if not np.array_equal(
                positions,
                np.arange(positions[0], positions[-1] + 1, dtype=np.int64),
            ):
                raise ValueError("episode teacher samples must be grouped by episode")
            indices = decision_indices[mask]
            if len(indices) == 0 or not np.array_equal(
                indices,
                np.arange(indices[0], indices[0] + len(indices), dtype=np.int64),
            ):
                raise ValueError("episode teacher decisions must be contiguous per episode")
        if isinstance(observations, Mapping) and "decision_index" in observations:
            observed = np.asarray(observations["decision_index"], dtype=np.int64).reshape(-1)
            if not np.array_equal(observed, decision_indices):
                raise ValueError("compact observation decision indices mismatch provenance")
        actions.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "decision_indices", decision_indices)
        object.__setattr__(self, "episode_ids", episode_ids)

    @property
    def episode_count(self) -> int:
        return int(np.unique(self.episode_ids).size)

    @property
    def decision_index_digest(self) -> str:
        return _array_digest(self.decision_indices)

    @property
    def episode_id_digest(self) -> str:
        return _array_digest(self.episode_ids)


@dataclass(frozen=True, slots=True)
class EpisodeTeacherArtifactManifest:
    artifact_digest: str
    arrays_digest: str
    observation_digest: str
    action_digest: str
    decision_index_digest: str
    episode_id_digest: str
    dataset_id: str
    train_start: int
    train_stop: int
    environment_digest: str
    action_spec_digest: str
    teacher_config_digest: str
    sample_count: int
    episode_count: int
    observation_keys: tuple[str, ...]
    observation_shapes: dict[str, tuple[int, ...]]
    observation_dtypes: dict[str, str]
    action_shape: tuple[int, int]
    schema_version: str = EPISODE_TEACHER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_digest", self.artifact_digest),
            ("arrays_digest", self.arrays_digest),
            ("observation_digest", self.observation_digest),
            ("action_digest", self.action_digest),
            ("decision_index_digest", self.decision_index_digest),
            ("episode_id_digest", self.episode_id_digest),
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("action_spec_digest", self.action_spec_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            require_sha256(value, field=name)
        if self.sample_count <= 0 or not 1 <= self.episode_count <= self.sample_count:
            raise ValueError("episode teacher manifest counts are invalid")
        if not self.observation_keys or tuple(sorted(self.observation_keys)) != (
            self.observation_keys
        ):
            raise ValueError("episode teacher observation keys are invalid")
        if set(self.observation_shapes) != set(self.observation_keys):
            raise ValueError("episode teacher observation shapes do not match keys")
        if set(self.observation_dtypes) != set(self.observation_keys):
            raise ValueError("episode teacher observation dtypes do not match keys")
        if any(
            shape[0] != self.sample_count for shape in self.observation_shapes.values()
        ):
            raise ValueError("episode teacher observation count mismatch")
        if self.action_shape[0] != self.sample_count:
            raise ValueError("episode teacher action count mismatch")
        if self.schema_version != EPISODE_TEACHER_ARTIFACT_SCHEMA:
            raise ValueError("unsupported episode teacher artifact schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_digest": self.action_digest,
            "action_shape": self.action_shape,
            "action_spec_digest": self.action_spec_digest,
            "arrays_digest": self.arrays_digest,
            "arrays_file": TEACHER_ARRAYS_NAME,
            "dataset_id": self.dataset_id,
            "decision_index_digest": self.decision_index_digest,
            "environment_digest": self.environment_digest,
            "episode_count": self.episode_count,
            "episode_id_digest": self.episode_id_digest,
            "observation_digest": self.observation_digest,
            "observation_dtypes": self.observation_dtypes,
            "observation_keys": self.observation_keys,
            "observation_shapes": self.observation_shapes,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "teacher_config_digest": self.teacher_config_digest,
            "train_start": self.train_start,
            "train_stop": self.train_stop,
        }


def write_episode_teacher_artifact(
    root: str | Path,
    dataset: EpisodeSupervisedPolicyDataset,
) -> str:
    output = Path(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"teacher artifact destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "actions": dataset.actions,
        "decision_indices": dataset.decision_indices,
        "episode_ids": dataset.episode_ids,
    }
    for key, value in _observation_arrays(dataset.observations).items():
        arrays[f"observation__{key}"] = value
    arrays_payload = _deterministic_npz(arrays)
    arrays_digest = _sha256(arrays_payload)
    base = {
        "action_digest": dataset.action_digest,
        "action_shape": dataset.actions.shape,
        "action_spec_digest": dataset.action_spec_digest,
        "arrays_digest": arrays_digest,
        "arrays_file": TEACHER_ARRAYS_NAME,
        "dataset_id": dataset.dataset_id,
        "decision_index_digest": dataset.decision_index_digest,
        "environment_digest": dataset.environment_digest,
        "episode_count": dataset.episode_count,
        "episode_id_digest": dataset.episode_id_digest,
        "observation_digest": dataset.observation_digest,
        "observation_dtypes": dataset.observation_dtypes,
        "observation_keys": dataset.observation_keys,
        "observation_shapes": dataset.observation_shapes,
        "sample_count": dataset.sample_count,
        "schema_version": EPISODE_TEACHER_ARTIFACT_SCHEMA,
        "teacher_config_digest": dataset.teacher_config_digest,
        "train_start": dataset.train_start,
        "train_stop": dataset.train_stop,
    }
    manifest = EpisodeTeacherArtifactManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        observation_digest=dataset.observation_digest,
        action_digest=dataset.action_digest,
        decision_index_digest=dataset.decision_index_digest,
        episode_id_digest=dataset.episode_id_digest,
        dataset_id=dataset.dataset_id,
        train_start=dataset.train_start,
        train_stop=dataset.train_stop,
        environment_digest=dataset.environment_digest,
        action_spec_digest=dataset.action_spec_digest,
        teacher_config_digest=dataset.teacher_config_digest,
        sample_count=dataset.sample_count,
        episode_count=dataset.episode_count,
        observation_keys=dataset.observation_keys,
        observation_shapes=dataset.observation_shapes,
        observation_dtypes=dataset.observation_dtypes,
        action_shape=(dataset.actions.shape[0], dataset.actions.shape[1]),
    )
    _atomic_write(output / TEACHER_ARRAYS_NAME, arrays_payload)
    _atomic_write(output / TEACHER_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest


def load_episode_teacher_artifact(
    root: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_environment_digest: str | None = None,
    expected_action_spec_digest: str | None = None,
    expected_train_range: tuple[int, int] | None = None,
) -> tuple[EpisodeTeacherArtifactManifest, EpisodeSupervisedPolicyDataset]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"teacher artifact directory is missing: {path}")
    entries = tuple(path.iterdir())
    if {entry.name for entry in entries} != _ALLOWED_FILES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("episode teacher artifact file closure mismatch")
    raw = json.loads((path / TEACHER_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("episode teacher manifest must be a mapping")
    try:
        manifest = EpisodeTeacherArtifactManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            observation_digest=str(raw["observation_digest"]),
            action_digest=str(raw["action_digest"]),
            decision_index_digest=str(raw["decision_index_digest"]),
            episode_id_digest=str(raw["episode_id_digest"]),
            dataset_id=str(raw["dataset_id"]),
            train_start=int(raw["train_start"]),
            train_stop=int(raw["train_stop"]),
            environment_digest=str(raw["environment_digest"]),
            action_spec_digest=str(raw["action_spec_digest"]),
            teacher_config_digest=str(raw["teacher_config_digest"]),
            sample_count=int(raw["sample_count"]),
            episode_count=int(raw["episode_count"]),
            observation_keys=tuple(str(value) for value in raw["observation_keys"]),
            observation_shapes={
                str(key): tuple(int(value) for value in shape)
                for key, shape in raw["observation_shapes"].items()
            },
            observation_dtypes={
                str(key): str(value) for key, value in raw["observation_dtypes"].items()
            },
            action_shape=tuple(int(value) for value in raw["action_shape"]),  # type: ignore[arg-type]
            schema_version=str(raw["schema_version"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("episode teacher artifact manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("episode teacher manifest digest mismatch")
    arrays_payload = (path / TEACHER_ARRAYS_NAME).read_bytes()
    if _sha256(arrays_payload) != manifest.arrays_digest:
        raise ValueError("episode teacher arrays digest mismatch")
    try:
        with np.load(io.BytesIO(arrays_payload), allow_pickle=False) as archive:
            expected_names = {"actions", "decision_indices", "episode_ids"} | {
                f"observation__{key}" for key in manifest.observation_keys
            }
            if set(archive.files) != expected_names:
                raise ValueError("episode teacher arrays names do not match contract")
            actions = archive["actions"]
            decision_indices = archive["decision_indices"]
            episode_ids = archive["episode_ids"]
            loaded = {
                key: archive[f"observation__{key}"] for key in manifest.observation_keys
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("episode teacher arrays are invalid") from error
    observations: ObservationBatch = (
        loaded["observations"]
        if manifest.observation_keys == ("observations",)
        else loaded
    )
    dataset = EpisodeSupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id=manifest.dataset_id,
        train_start=manifest.train_start,
        train_stop=manifest.train_stop,
        environment_digest=manifest.environment_digest,
        action_spec_digest=manifest.action_spec_digest,
        teacher_config_digest=manifest.teacher_config_digest,
        decision_indices=decision_indices,
        episode_ids=episode_ids,
    )
    if (
        dataset.observation_digest != manifest.observation_digest
        or dataset.action_digest != manifest.action_digest
        or dataset.decision_index_digest != manifest.decision_index_digest
        or dataset.episode_id_digest != manifest.episode_id_digest
        or dataset.episode_count != manifest.episode_count
    ):
        raise ValueError("episode teacher artifact content digest mismatch")
    if dataset.observation_shapes != manifest.observation_shapes:
        raise ValueError("episode teacher observation shape mismatch")
    if dataset.observation_dtypes != manifest.observation_dtypes:
        raise ValueError("episode teacher observation dtype mismatch")
    if expected_dataset_id is not None and dataset.dataset_id != expected_dataset_id:
        raise ValueError("episode teacher dataset identity mismatch")
    if (
        expected_environment_digest is not None
        and dataset.environment_digest != expected_environment_digest
    ):
        raise ValueError("episode teacher environment identity mismatch")
    if (
        expected_action_spec_digest is not None
        and dataset.action_spec_digest != expected_action_spec_digest
    ):
        raise ValueError("episode teacher action specification identity mismatch")
    if expected_train_range is not None and (
        dataset.train_start,
        dataset.train_stop,
    ) != expected_train_range:
        raise ValueError("episode teacher training envelope mismatch")
    return manifest, dataset


def _append_observation(
    observation: ObservationValue,
    *,
    expected_index: int,
    flat: list[np.ndarray],
    structured: dict[str, list[np.ndarray]] | None,
    expected_keys: tuple[str, ...] | None,
    expected_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict[str, list[np.ndarray]] | None, tuple[str, ...] | None]:
    if isinstance(observation, Mapping):
        keys = tuple(sorted(observation))
        if expected_keys is None:
            expected_keys = keys
            missing = set(_COMPACT_KEYS) - set(keys)
            if missing:
                raise ValueError(
                    f"structured teacher observation is missing compact keys: {sorted(missing)}"
                )
            structured = {key: [] for key in _COMPACT_KEYS}
            structured["decision_index"] = []
            expected_shapes.update(
                {key: np.asarray(observation[key]).shape for key in keys}
            )
        if keys != expected_keys or structured is None:
            raise ValueError("teacher structured observation keys changed during rollout")
        for key in keys:
            if np.asarray(observation[key]).shape != expected_shapes[key]:
                raise ValueError("teacher observation shape changed during rollout")
        for key in _COMPACT_KEYS:
            structured[key].append(np.asarray(observation[key]).copy(order="C"))
        structured["decision_index"].append(
            np.asarray([expected_index], dtype=np.int64)
        )
    elif expected_keys is not None:
        raise ValueError("teacher observation kind changed during rollout")
    else:
        flat.append(np.asarray(observation, dtype=np.float32).copy(order="C"))
    return structured, expected_keys


def collect_episode_teacher_rollout(
    environment: TeacherRolloutEnvironment,
    batch: EpisodeOracleBatch,
    *,
    teacher_config_digest: str,
) -> EpisodeSupervisedPolicyDataset:
    """Roll out each sampled episode independently with its declared reset state."""

    require_sha256(teacher_config_digest, field="teacher_config_digest")
    flat_observations: list[np.ndarray] = []
    structured_observations: dict[str, list[np.ndarray]] | None = None
    expected_keys: tuple[str, ...] | None = None
    expected_shapes: dict[str, tuple[int, ...]] = {}
    actions: list[np.ndarray] = []
    decision_indices: list[int] = []
    episode_ids: list[int] = []
    for contract, targets in zip(batch.contracts, batch.targets, strict=True):
        expected_count = contract.stop - contract.start - 1
        observation, info = environment.reset(
            options={
                "start_idx": contract.start,
                "episode_bars": expected_count,
                "initial_state_mode": contract.initial_state_mode,
            }
        )
        if int(info.get("start_index", -1)) != contract.start:
            raise ValueError("episode teacher environment reset start mismatch")
        if isinstance(observation, Mapping) and "current_weights" in observation:
            actual = np.asarray(observation["current_weights"], dtype=np.float64)
            if not np.allclose(actual, contract.initial_weights, atol=1e-12, rtol=0.0):
                raise ValueError("episode teacher reset weights mismatch contract")
        for offset, target in enumerate(targets):
            expected_index = contract.start + offset
            if environment.current_index != expected_index:
                raise ValueError("episode teacher environment left its contract")
            structured_observations, expected_keys = _append_observation(
                observation,
                expected_index=expected_index,
                flat=flat_observations,
                structured=structured_observations,
                expected_keys=expected_keys,
                expected_shapes=expected_shapes,
            )
            actions.append(np.asarray(target, dtype=np.float32).copy(order="C"))
            decision_indices.append(expected_index)
            episode_ids.append(contract.episode_index)
            observation, _, terminated, truncated, _ = environment.step(target)
            if (terminated or truncated) != (offset == expected_count - 1):
                raise ValueError("episode teacher environment ended outside its contract")
    observations: ObservationBatch
    if structured_observations is not None:
        observations = {
            key: np.stack(values, axis=0)
            for key, values in structured_observations.items()
        }
    else:
        observations = np.asarray(flat_observations, dtype=np.float32)
    return EpisodeSupervisedPolicyDataset(
        observations=observations,
        actions=np.stack(actions, axis=0),
        dataset_id=batch.dataset_id,
        train_start=min(contract.start for contract in batch.contracts),
        train_stop=max(contract.stop for contract in batch.contracts),
        environment_digest=environment.environment_digest,
        action_spec_digest=environment.action_spec_digest,
        teacher_config_digest=teacher_config_digest,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


__all__ = [
    "EPISODE_TEACHER_ARTIFACT_SCHEMA",
    "EpisodeSupervisedPolicyDataset",
    "EpisodeTeacherArtifactManifest",
    "collect_episode_teacher_rollout",
    "load_episode_teacher_artifact",
    "write_episode_teacher_artifact",
]
