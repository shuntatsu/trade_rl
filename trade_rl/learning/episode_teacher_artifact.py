"""Immutable episode-aligned supervised datasets for Oracle behavior cloning."""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.learning.oracle_bellman_contracts import OracleSolverProvenance
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
    _sha256,
)

EPISODE_TEACHER_ARTIFACT_SCHEMA_V1: Final = "episode_supervised_teacher_artifact_v1"
EPISODE_TEACHER_ARTIFACT_SCHEMA_V2: Final = "episode_supervised_teacher_artifact_v2"
EPISODE_TEACHER_ARTIFACT_SCHEMA: Final = "episode_supervised_teacher_artifact_v3"
_ALLOWED_FILES = frozenset({TEACHER_MANIFEST_NAME, TEACHER_ARRAYS_NAME})
_COMPACT_KEYS = (
    "active",
    "asset_state",
    "current_snapshot",
    "current_weights",
    "global_state",
)

_MAX_EPISODES_PER_TEACHER_TASK: Final = 8
_FORK_EPISODE_BATCH: EpisodeOracleBatch | None = None
_FORK_EPISODE_ENVIRONMENT_FACTORY: Any | None = None
_FORK_EPISODE_TEACHER_DIGEST: str | None = None


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
    solver_provenance: OracleSolverProvenance | None = None

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
            raise ValueError(
                "episode teacher actions must be a non-empty finite matrix"
            )
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
            raise ValueError(
                "episode teacher decision indices leave the train envelope"
            )
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
            if len(indices) == 0 or (
                len(indices) > 1 and np.any(np.diff(indices) <= 0)
            ):
                raise ValueError(
                    "episode teacher decisions must be strictly increasing per episode"
                )
        if isinstance(observations, Mapping) and "decision_index" in observations:
            observed = np.asarray(
                observations["decision_index"], dtype=np.int64
            ).reshape(-1)
            if not np.array_equal(observed, decision_indices):
                raise ValueError(
                    "compact observation decision indices mismatch provenance"
                )
        if self.solver_provenance is not None and not isinstance(
            self.solver_provenance, OracleSolverProvenance
        ):
            raise ValueError("solver_provenance must be OracleSolverProvenance")
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
    solver_provenance_digest: str | None = None
    solver_provenance: OracleSolverProvenance | None = None
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
        if self.schema_version not in {
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V2,
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
        }:
            raise ValueError("unsupported episode teacher artifact schema")
        if self.schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA_V1:
            if (
                self.solver_provenance is not None
                or self.solver_provenance_digest is not None
            ):
                raise ValueError(
                    "legacy episode teacher artifacts cannot claim provenance"
                )
        elif not isinstance(self.solver_provenance, OracleSolverProvenance):
            raise ValueError("episode teacher artifacts require solver provenance")
        elif self.schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA_V2:
            if self.solver_provenance_digest is not None:
                raise ValueError(
                    "v2 episode teacher artifacts cannot claim v3 integrity"
                )
        else:
            if self.solver_provenance_digest is None:
                raise ValueError(
                    "v3 episode teacher artifacts require provenance digest"
                )
            require_sha256(
                self.solver_provenance_digest,
                field="solver_provenance_digest",
            )
            expected_provenance_digest = content_digest(
                self.solver_provenance.serialized_payload()
            )
            if self.solver_provenance_digest != expected_provenance_digest:
                raise ValueError("episode teacher manifest provenance digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        """Return numerical artifact identity without volatile runtime evidence."""

        payload: dict[str, object] = {
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
        if self.schema_version in {
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V2,
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
        }:
            if self.solver_provenance is None:  # pragma: no cover - guarded above
                raise RuntimeError("solver provenance disappeared")
            payload["solver_identity"] = {
                **self.solver_provenance.identity_payload(),
                "digest": self.solver_provenance.digest,
            }
        return payload

    def serialized_payload(self) -> dict[str, object]:
        """Return complete manifest with runtime evidence outside artifact identity."""

        payload = self.digest_payload()
        if self.schema_version in {
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V2,
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
        }:
            if self.solver_provenance is None:  # pragma: no cover - guarded above
                raise RuntimeError("solver provenance disappeared")
            payload["solver_provenance"] = self.solver_provenance.serialized_payload()
        if self.schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA:
            payload["solver_provenance_digest"] = self.solver_provenance_digest
        return {**payload, "artifact_digest": self.artifact_digest}


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
    schema_version = (
        EPISODE_TEACHER_ARTIFACT_SCHEMA_V1
        if dataset.solver_provenance is None
        else EPISODE_TEACHER_ARTIFACT_SCHEMA
    )
    solver_provenance_digest = (
        None
        if dataset.solver_provenance is None
        else content_digest(dataset.solver_provenance.serialized_payload())
    )
    base: dict[str, object] = {
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
        "schema_version": schema_version,
        "teacher_config_digest": dataset.teacher_config_digest,
        "train_start": dataset.train_start,
        "train_stop": dataset.train_stop,
    }
    if dataset.solver_provenance is not None:
        base["solver_identity"] = {
            **dataset.solver_provenance.identity_payload(),
            "digest": dataset.solver_provenance.digest,
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
        solver_provenance_digest=solver_provenance_digest,
        solver_provenance=dataset.solver_provenance,
        schema_version=schema_version,
    )
    _atomic_write(output / TEACHER_ARRAYS_NAME, arrays_payload)
    _atomic_write(
        output / TEACHER_MANIFEST_NAME,
        canonical_json_bytes(manifest.serialized_payload()),
    )
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
        schema_version = str(raw["schema_version"])
        solver_provenance = (
            None
            if schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA_V1
            else OracleSolverProvenance.from_payload(raw["solver_provenance"])
        )
        solver_provenance_digest = (
            str(raw["solver_provenance_digest"])
            if schema_version == EPISODE_TEACHER_ARTIFACT_SCHEMA
            else None
        )
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
            solver_provenance_digest=solver_provenance_digest,
            solver_provenance=solver_provenance,
            schema_version=schema_version,
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
        solver_provenance=manifest.solver_provenance,
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
    if (
        expected_train_range is not None
        and (dataset.train_start, dataset.train_stop) != expected_train_range
    ):
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
            raise ValueError(
                "teacher structured observation keys changed during rollout"
            )
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
        raw_start = info.get("start_index")
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, int)
            or raw_start != contract.start
        ):
            raise ValueError("episode teacher environment reset start mismatch")
        if isinstance(observation, Mapping) and "current_weights" in observation:
            actual = np.asarray(observation["current_weights"], dtype=np.float64)
            if not np.allclose(actual, contract.initial_weights, atol=1e-6, rtol=0.0):
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
                raise ValueError(
                    "episode teacher environment ended outside its contract"
                )
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
        solver_provenance=batch.solver_provenance,
    )


def _collect_episode_with_environment(
    environment: TeacherRolloutEnvironment,
    batch: EpisodeOracleBatch,
    teacher_config_digest: str,
    item: tuple[OracleEpisodeContract, np.ndarray],
) -> tuple[EpisodeSupervisedPolicyDataset, int]:
    contract, targets = item
    isolated_contract = OracleEpisodeContract(
        dataset_id=contract.dataset_id,
        episode_index=0,
        start=contract.start,
        stop=contract.stop,
        initial_state_mode=contract.initial_state_mode,
        initial_weights=contract.initial_weights,
    )
    episode_batch = EpisodeOracleBatch(
        dataset_id=batch.dataset_id,
        teacher_config_digest=batch.teacher_config_digest,
        sampling_config_digest=batch.sampling_config_digest,
        contracts=(isolated_contract,),
        targets=(targets,),
        solver_provenance=batch.solver_provenance,
    )
    episode = collect_episode_teacher_rollout(
        environment,
        episode_batch,
        teacher_config_digest=teacher_config_digest,
    )
    return episode, contract.episode_index


def _collect_isolated_episode_chunk(
    environment_factory: Any,
    batch: EpisodeOracleBatch,
    teacher_config_digest: str,
    items: tuple[tuple[OracleEpisodeContract, np.ndarray], ...],
) -> tuple[tuple[EpisodeSupervisedPolicyDataset, int], ...]:
    if not items:
        raise ValueError("teacher rollout episode chunk must not be empty")
    environment = environment_factory()
    try:
        return tuple(
            _collect_episode_with_environment(
                environment,
                batch,
                teacher_config_digest,
                item,
            )
            for item in items
        )
    finally:
        environment.close()


def _episode_item_chunks(
    items: tuple[tuple[OracleEpisodeContract, np.ndarray], ...],
    *,
    worker_count: int,
) -> tuple[tuple[tuple[OracleEpisodeContract, np.ndarray], ...], ...]:
    if not items:
        return ()
    if worker_count <= 0:
        raise ValueError("teacher rollout chunk worker count must be positive")
    minimum_task_count = (
        len(items) + _MAX_EPISODES_PER_TEACHER_TASK - 1
    ) // _MAX_EPISODES_PER_TEACHER_TASK
    task_count = max(min(worker_count, len(items)), minimum_task_count)
    base_size, oversized_chunk_count = divmod(len(items), task_count)
    chunks: list[tuple[tuple[OracleEpisodeContract, np.ndarray], ...]] = []
    offset = 0
    for chunk_index in range(task_count):
        chunk_size = base_size + int(chunk_index < oversized_chunk_count)
        chunks.append(tuple(items[offset : offset + chunk_size]))
        offset += chunk_size
    return tuple(chunks)


def _collect_forked_episode_chunk(
    items: tuple[tuple[OracleEpisodeContract, np.ndarray], ...],
) -> tuple[tuple[EpisodeSupervisedPolicyDataset, int], ...]:
    if (
        _FORK_EPISODE_ENVIRONMENT_FACTORY is None
        or _FORK_EPISODE_BATCH is None
        or _FORK_EPISODE_TEACHER_DIGEST is None
    ):
        raise RuntimeError("forked episode teacher worker is not initialized")
    return _collect_isolated_episode_chunk(
        _FORK_EPISODE_ENVIRONMENT_FACTORY,
        _FORK_EPISODE_BATCH,
        _FORK_EPISODE_TEACHER_DIGEST,
        items,
    )


def collect_episode_teacher_rollout_parallel(
    environment_factory: Any,
    batch: EpisodeOracleBatch,
    *,
    teacher_config_digest: str,
    max_workers: int,
    shard_root: str | Path | None = None,
) -> EpisodeSupervisedPolicyDataset:
    """Collect independent teacher episodes concurrently and preserve serial order."""

    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ValueError("teacher rollout worker count must be an integer")
    if max_workers <= 0:
        raise ValueError("teacher rollout worker count must be positive")
    if max_workers == 1 or batch.episode_count == 1:
        environment = environment_factory()
        try:
            return collect_episode_teacher_rollout(
                environment,
                batch,
                teacher_config_digest=teacher_config_digest,
            )
        finally:
            environment.close()

    worker_count = min(max_workers, batch.episode_count)
    items = tuple(zip(batch.contracts, batch.targets, strict=True))
    resolved_shard_root = None if shard_root is None else Path(shard_root)
    collected_by_id: dict[int, tuple[EpisodeSupervisedPolicyDataset, int]] = {}
    pending_items: list[tuple[OracleEpisodeContract, np.ndarray]] = []
    for item in items:
        contract, _ = item
        shard_path = (
            None
            if resolved_shard_root is None
            else resolved_shard_root / contract.digest
        )
        if shard_path is None or not shard_path.exists():
            pending_items.append(item)
            continue
        _, episode = load_episode_teacher_artifact(
            shard_path,
            expected_dataset_id=batch.dataset_id,
            expected_train_range=(contract.start, contract.stop),
        )
        if episode.teacher_config_digest != teacher_config_digest:
            raise ValueError("cached episode teacher shard identity mismatch")
        collected_by_id[contract.episode_index] = (
            episode,
            contract.episode_index,
        )

    def persist(
        value: tuple[EpisodeSupervisedPolicyDataset, int],
    ) -> None:
        episode, episode_id = value
        collected_by_id[episode_id] = value
        if resolved_shard_root is None:
            return
        contract = batch.contracts[episode_id]
        shard_path = resolved_shard_root / contract.digest
        if shard_path.exists():
            return
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{shard_path.name}.",
                dir=str(shard_path.parent),
            )
        )
        try:
            write_episode_teacher_artifact(temporary, episode)
            try:
                temporary.replace(shard_path)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    if pending_items:
        pending_worker_count = min(worker_count, len(pending_items))
        pending_chunks = _episode_item_chunks(
            tuple(pending_items),
            worker_count=pending_worker_count,
        )
    else:
        pending_worker_count = 0
        pending_chunks = ()

    if pending_chunks and "fork" in mp.get_all_start_methods():
        global _FORK_EPISODE_BATCH
        global _FORK_EPISODE_ENVIRONMENT_FACTORY
        global _FORK_EPISODE_TEACHER_DIGEST
        _FORK_EPISODE_BATCH = batch
        _FORK_EPISODE_ENVIRONMENT_FACTORY = environment_factory
        _FORK_EPISODE_TEACHER_DIGEST = teacher_config_digest
        try:
            context = mp.get_context("fork")
            with context.Pool(
                processes=pending_worker_count,
                maxtasksperchild=1,
            ) as pool:
                for values in pool.imap(
                    _collect_forked_episode_chunk,
                    pending_chunks,
                    chunksize=1,
                ):
                    for value in values:
                        persist(value)
        finally:
            _FORK_EPISODE_BATCH = None
            _FORK_EPISODE_ENVIRONMENT_FACTORY = None
            _FORK_EPISODE_TEACHER_DIGEST = None
    elif pending_chunks:

        def collect_chunk(
            items: tuple[tuple[OracleEpisodeContract, np.ndarray], ...],
        ) -> tuple[tuple[EpisodeSupervisedPolicyDataset, int], ...]:
            return _collect_isolated_episode_chunk(
                environment_factory,
                batch,
                teacher_config_digest,
                items,
            )

        with ThreadPoolExecutor(
            max_workers=pending_worker_count,
            thread_name_prefix="teacher-rollout",
        ) as executor:
            for values in executor.map(collect_chunk, pending_chunks):
                for value in values:
                    persist(value)
    collected = tuple(
        collected_by_id[contract.episode_index] for contract in batch.contracts
    )
    episodes = tuple(item[0] for item in collected)

    first = episodes[0]
    if any(
        episode.dataset_id != first.dataset_id
        or episode.environment_digest != first.environment_digest
        or episode.action_spec_digest != first.action_spec_digest
        or episode.teacher_config_digest != first.teacher_config_digest
        or (
            None
            if episode.solver_provenance is None
            else episode.solver_provenance.digest
        )
        != (None if first.solver_provenance is None else first.solver_provenance.digest)
        or isinstance(episode.observations, Mapping)
        != isinstance(first.observations, Mapping)
        for episode in episodes[1:]
    ):
        raise ValueError("parallel teacher episode identities do not match")
    if isinstance(first.observations, Mapping):
        keys = tuple(sorted(first.observations))
        if any(tuple(sorted(episode.observations)) != keys for episode in episodes):
            raise ValueError("parallel teacher observation keys do not match")
        observations: ObservationBatch = {
            key: np.concatenate(
                [np.asarray(episode.observations[key]) for episode in episodes],
                axis=0,
            )
            for key in keys
        }
    else:
        observations = np.concatenate(
            [np.asarray(episode.observations) for episode in episodes], axis=0
        )
    return EpisodeSupervisedPolicyDataset(
        observations=observations,
        actions=np.concatenate([episode.actions for episode in episodes], axis=0),
        dataset_id=first.dataset_id,
        train_start=min(episode.train_start for episode in episodes),
        train_stop=max(episode.train_stop for episode in episodes),
        environment_digest=first.environment_digest,
        action_spec_digest=first.action_spec_digest,
        teacher_config_digest=first.teacher_config_digest,
        decision_indices=np.concatenate(
            [episode.decision_indices for episode in episodes], axis=0
        ),
        episode_ids=np.concatenate(
            [
                np.full(episode.sample_count, episode_id, dtype=np.int64)
                for episode, episode_id in collected
            ]
        ),
        solver_provenance=first.solver_provenance,
    )


__all__ = [
    "EPISODE_TEACHER_ARTIFACT_SCHEMA",
    "EPISODE_TEACHER_ARTIFACT_SCHEMA_V1",
    "EpisodeSupervisedPolicyDataset",
    "EpisodeTeacherArtifactManifest",
    "collect_episode_teacher_rollout",
    "collect_episode_teacher_rollout_parallel",
    "load_episode_teacher_artifact",
    "write_episode_teacher_artifact",
]
