"""Immutable supervised-policy datasets for oracle behavior cloning."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

TEACHER_ARTIFACT_SCHEMA: Final = "supervised_policy_teacher_artifact_v1"
TEACHER_MANIFEST_NAME: Final = "manifest.json"
TEACHER_ARRAYS_NAME: Final = "arrays.npz"
_ALLOWED_FILES = frozenset({TEACHER_MANIFEST_NAME, TEACHER_ARRAYS_NAME})
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _array_digest(array: np.ndarray) -> str:
    return _sha256(_npy_bytes(np.asarray(array)))


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class SupervisedPolicyDataset:
    observations: np.ndarray
    actions: np.ndarray
    dataset_id: str
    train_start: int
    train_stop: int
    environment_digest: str
    action_spec_digest: str
    teacher_config_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("action_spec_digest", self.action_spec_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            require_sha256(value, field=name)
        observations = np.asarray(self.observations, dtype=np.float32).copy(order="C")
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError("teacher observations and actions must be rank-two")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("teacher observations and actions must be finite")
        expected_count = self.train_stop - self.train_start - 1
        if expected_count <= 0 or len(observations) != expected_count:
            raise ValueError("teacher sample count must cover every training decision")
        if len(actions) != expected_count:
            raise ValueError("teacher action sample count does not match observations")
        observations.setflags(write=False)
        actions.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)

    @property
    def observation_digest(self) -> str:
        return _array_digest(self.observations)

    @property
    def action_digest(self) -> str:
        return _array_digest(self.actions)


@dataclass(frozen=True, slots=True)
class TeacherArtifactManifest:
    artifact_digest: str
    arrays_digest: str
    observation_digest: str
    action_digest: str
    dataset_id: str
    train_start: int
    train_stop: int
    environment_digest: str
    action_spec_digest: str
    teacher_config_digest: str
    sample_count: int
    observation_shape: tuple[int, int]
    action_shape: tuple[int, int]
    schema_version: str = TEACHER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_digest", self.artifact_digest),
            ("arrays_digest", self.arrays_digest),
            ("observation_digest", self.observation_digest),
            ("action_digest", self.action_digest),
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("action_spec_digest", self.action_spec_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            require_sha256(value, field=name)
        if self.sample_count <= 0:
            raise ValueError("teacher artifact sample_count must be positive")
        if self.observation_shape[0] != self.sample_count:
            raise ValueError("teacher observation shape does not match sample_count")
        if self.action_shape[0] != self.sample_count:
            raise ValueError("teacher action shape does not match sample_count")
        if self.schema_version != TEACHER_ARTIFACT_SCHEMA:
            raise ValueError("unsupported teacher artifact schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_digest": self.action_digest,
            "action_shape": self.action_shape,
            "action_spec_digest": self.action_spec_digest,
            "arrays_digest": self.arrays_digest,
            "arrays_file": TEACHER_ARRAYS_NAME,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "observation_digest": self.observation_digest,
            "observation_shape": self.observation_shape,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "teacher_config_digest": self.teacher_config_digest,
            "train_start": self.train_start,
            "train_stop": self.train_stop,
        }


def write_teacher_artifact(
    root: str | Path,
    dataset: SupervisedPolicyDataset,
) -> str:
    output = Path(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"teacher artifact destination is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arrays_payload = _deterministic_npz(
        {"actions": dataset.actions, "observations": dataset.observations}
    )
    arrays_digest = _sha256(arrays_payload)
    base = {
        "action_digest": dataset.action_digest,
        "action_shape": dataset.actions.shape,
        "action_spec_digest": dataset.action_spec_digest,
        "arrays_digest": arrays_digest,
        "arrays_file": TEACHER_ARRAYS_NAME,
        "dataset_id": dataset.dataset_id,
        "environment_digest": dataset.environment_digest,
        "observation_digest": dataset.observation_digest,
        "observation_shape": dataset.observations.shape,
        "sample_count": len(dataset.observations),
        "schema_version": TEACHER_ARTIFACT_SCHEMA,
        "teacher_config_digest": dataset.teacher_config_digest,
        "train_start": dataset.train_start,
        "train_stop": dataset.train_stop,
    }
    manifest = TeacherArtifactManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        observation_digest=dataset.observation_digest,
        action_digest=dataset.action_digest,
        dataset_id=dataset.dataset_id,
        train_start=dataset.train_start,
        train_stop=dataset.train_stop,
        environment_digest=dataset.environment_digest,
        action_spec_digest=dataset.action_spec_digest,
        teacher_config_digest=dataset.teacher_config_digest,
        sample_count=len(dataset.observations),
        observation_shape=(dataset.observations.shape[0], dataset.observations.shape[1]),
        action_shape=(dataset.actions.shape[0], dataset.actions.shape[1]),
    )
    _atomic_write(output / TEACHER_ARRAYS_NAME, arrays_payload)
    _atomic_write(output / TEACHER_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest


def _load_manifest(root: Path) -> TeacherArtifactManifest:
    if not root.is_dir():
        raise FileNotFoundError(f"teacher artifact directory is missing: {root}")
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != _ALLOWED_FILES or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("teacher artifact file closure mismatch")
    raw = json.loads((root / TEACHER_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("teacher artifact manifest must be a mapping")
    try:
        manifest = TeacherArtifactManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            observation_digest=str(raw["observation_digest"]),
            action_digest=str(raw["action_digest"]),
            dataset_id=str(raw["dataset_id"]),
            train_start=int(raw["train_start"]),
            train_stop=int(raw["train_stop"]),
            environment_digest=str(raw["environment_digest"]),
            action_spec_digest=str(raw["action_spec_digest"]),
            teacher_config_digest=str(raw["teacher_config_digest"]),
            sample_count=int(raw["sample_count"]),
            observation_shape=tuple(int(value) for value in raw["observation_shape"]),  # type: ignore[arg-type]
            action_shape=tuple(int(value) for value in raw["action_shape"]),  # type: ignore[arg-type]
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("teacher artifact manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("teacher manifest digest mismatch")
    return manifest


def load_teacher_artifact(
    root: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_environment_digest: str | None = None,
    expected_action_spec_digest: str | None = None,
    expected_train_range: tuple[int, int] | None = None,
) -> tuple[TeacherArtifactManifest, SupervisedPolicyDataset]:
    path = Path(root)
    manifest = _load_manifest(path)
    arrays_payload = (path / TEACHER_ARRAYS_NAME).read_bytes()
    if _sha256(arrays_payload) != manifest.arrays_digest:
        raise ValueError("teacher arrays digest mismatch")
    try:
        with np.load(io.BytesIO(arrays_payload), allow_pickle=False) as archive:
            if set(archive.files) != {"actions", "observations"}:
                raise ValueError("teacher arrays names do not match contract")
            actions = archive["actions"]
            observations = archive["observations"]
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("teacher arrays are invalid") from error
    dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id=manifest.dataset_id,
        train_start=manifest.train_start,
        train_stop=manifest.train_stop,
        environment_digest=manifest.environment_digest,
        action_spec_digest=manifest.action_spec_digest,
        teacher_config_digest=manifest.teacher_config_digest,
    )
    if dataset.observation_digest != manifest.observation_digest:
        raise ValueError("teacher observation digest mismatch")
    if dataset.action_digest != manifest.action_digest:
        raise ValueError("teacher action digest mismatch")
    if expected_dataset_id is not None and dataset.dataset_id != expected_dataset_id:
        raise ValueError("teacher dataset identity mismatch")
    if (
        expected_environment_digest is not None
        and dataset.environment_digest != expected_environment_digest
    ):
        raise ValueError("teacher environment identity mismatch")
    if (
        expected_action_spec_digest is not None
        and dataset.action_spec_digest != expected_action_spec_digest
    ):
        raise ValueError("teacher action specification identity mismatch")
    if expected_train_range is not None and (
        dataset.train_start,
        dataset.train_stop,
    ) != expected_train_range:
        raise ValueError("teacher training range identity mismatch")
    return manifest, dataset


class TeacherRolloutEnvironment(Protocol):
    current_index: int

    @property
    def environment_digest(self) -> str: ...

    @property
    def action_spec_digest(self) -> str: ...

    def reset(
        self,
        *,
        options: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, object]]: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...


def collect_teacher_rollout(
    environment: TeacherRolloutEnvironment,
    targets: np.ndarray,
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    teacher_config_digest: str,
) -> SupervisedPolicyDataset:
    """Execute every train-range oracle action and pair it with causal observations."""

    start, stop = train_range
    action_array = np.asarray(targets, dtype=np.float32)
    expected_count = stop - start - 1
    if action_array.ndim != 2 or len(action_array) != expected_count:
        raise ValueError("oracle targets do not cover the exact training range")
    observation, _ = environment.reset(
        options={
            "start_idx": start,
            "episode_bars": expected_count,
            "initial_state_mode": "cash",
        }
    )
    observations: list[np.ndarray] = []
    for offset, target in enumerate(action_array):
        expected_index = start + offset
        if environment.current_index != expected_index:
            raise ValueError("teacher environment did not advance one training bar")
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        observation, _, terminated, truncated, _ = environment.step(target)
        if (terminated or truncated) != (offset == expected_count - 1):
            raise ValueError("teacher environment ended outside the training range")
    return SupervisedPolicyDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=action_array,
        dataset_id=dataset_id,
        train_start=start,
        train_stop=stop,
        environment_digest=environment.environment_digest,
        action_spec_digest=environment.action_spec_digest,
        teacher_config_digest=teacher_config_digest,
    )


__all__ = [
    "TEACHER_ARRAYS_NAME",
    "TEACHER_ARTIFACT_SCHEMA",
    "TEACHER_MANIFEST_NAME",
    "SupervisedPolicyDataset",
    "TeacherArtifactManifest",
    "TeacherRolloutEnvironment",
    "collect_teacher_rollout",
    "load_teacher_artifact",
    "write_teacher_artifact",
]
