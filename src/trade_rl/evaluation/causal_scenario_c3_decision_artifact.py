"""Deterministic persisted-before-replay artifacts for C3 decisions."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3_DECISION_SCHEMA,
    PersistedScenarioDecision,
)

C3_DECISION_ARTIFACT_SCHEMA: Final = "causal_scenario_c3_decision_artifact_v1"
_MANIFEST_NAME: Final = "decision.json"
_ARRAYS_NAME: Final = "arrays.npz"
_ALLOWED_FILES: Final = frozenset({_MANIFEST_NAME, _ARRAYS_NAME})
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_exact_files(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"C3 decision artifact directory is missing: {root}")
    actual: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("C3 decision artifact contains invalid file entry")
        actual.add(entry.name)
    if actual != _ALLOWED_FILES:
        raise ValueError("C3 decision artifact file closure mismatch")


def _arrays(decision: PersistedScenarioDecision) -> dict[str, np.ndarray]:
    return {
        "projected_targets": np.asarray(decision.projected_targets),
        "raw_candidate_actions": np.asarray(decision.raw_candidate_actions),
        "regret": np.asarray(decision.regret),
        "score": np.asarray(decision.score),
        "tie_candidate_indices": np.asarray(
            decision.tie_candidate_indices, dtype=np.int64
        ),
    }


def _array_metadata(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": np.asarray(array).dtype.str,
            "shape": tuple(int(size) for size in np.asarray(array).shape),
        }
        for name, array in sorted(arrays.items())
    }


def _base_manifest(
    decision: PersistedScenarioDecision,
    *,
    arrays_digest: str,
    metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "action_spec_digest": decision.action_spec_digest,
        "array_metadata": metadata,
        "arrays_digest": arrays_digest,
        "arrays_file": _ARRAYS_NAME,
        "candidate_digests": decision.candidate_digests,
        "candidate_generator_digest": decision.candidate_generator_digest,
        "created_before_realized_replay": decision.created_before_realized_replay,
        "dataset_id": decision.dataset_id,
        "decision_digest": decision.decision_digest,
        "decision_schema_version": decision.schema_version,
        "environment_digest": decision.environment_digest,
        "execution_policy_digest": decision.execution_policy_digest,
        "fold_digest": decision.fold_digest,
        "observation_digest": decision.observation_digest,
        "query_index": decision.query_index,
        "query_timestamp_ns": decision.query_timestamp_ns,
        "realized_stop_index": decision.realized_stop_index,
        "risk_digest": decision.risk_digest,
        "scenario_library_digest": decision.scenario_library_digest,
        "scenario_set_digest": decision.scenario_set_digest,
        "schema_version": C3_DECISION_ARTIFACT_SCHEMA,
        "selected_candidate_digest": decision.selected_candidate_digest,
        "selected_candidate_index": decision.selected_candidate_index,
        "starting_equity": decision.starting_equity,
        "state_snapshot_digest": decision.state_snapshot_digest,
        "value_result_digest": decision.value_result_digest,
        "zero_candidate_index": decision.zero_candidate_index,
    }


@dataclass(frozen=True, slots=True)
class LoadedC3Decision:
    decision: PersistedScenarioDecision
    artifact_digest: str
    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PersistedScenarioDecision):
            raise ValueError("decision must be PersistedScenarioDecision")
        object.__setattr__(
            self,
            "artifact_digest",
            require_sha256(self.artifact_digest, field="artifact_digest"),
        )
        object.__setattr__(self, "root", Path(self.root))


def write_c3_decision_artifact(
    root: str | Path, decision: PersistedScenarioDecision
) -> str:
    destination = Path(root)
    if destination.exists() and any(destination.iterdir()):
        loaded = load_c3_decision_artifact(destination)
        if loaded.decision.decision_digest != decision.decision_digest:
            raise FileExistsError(
                f"conflicting C3 decision artifact already exists: {destination}"
            )
        return loaded.artifact_digest

    arrays = _arrays(decision)
    arrays_payload = _deterministic_npz(arrays)
    base = _base_manifest(
        decision,
        arrays_digest=_sha256_bytes(arrays_payload),
        metadata=_array_metadata(arrays),
    )
    artifact_digest = content_digest(base)
    manifest = dict(base)
    manifest["artifact_digest"] = artifact_digest
    manifest_payload = canonical_json_bytes(manifest)

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination / _ARRAYS_NAME, arrays_payload)
    _atomic_write(destination / _MANIFEST_NAME, manifest_payload)
    loaded = load_c3_decision_artifact(destination)
    if loaded.artifact_digest != artifact_digest:
        raise ValueError("published C3 decision artifact failed verification")
    return artifact_digest


def _expect_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _expect_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _expect_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _expect_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def load_c3_decision_artifact(root: str | Path) -> LoadedC3Decision:
    source = Path(root)
    _verify_exact_files(source)
    manifest_path = source / _MANIFEST_NAME
    arrays_path = source / _ARRAYS_NAME
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C3 decision manifest is invalid") from error
    manifest = _expect_mapping(manifest_raw, field="manifest")
    expected_fields = {
        "action_spec_digest",
        "array_metadata",
        "arrays_digest",
        "arrays_file",
        "artifact_digest",
        "candidate_digests",
        "candidate_generator_digest",
        "created_before_realized_replay",
        "dataset_id",
        "decision_digest",
        "decision_schema_version",
        "environment_digest",
        "execution_policy_digest",
        "fold_digest",
        "observation_digest",
        "query_index",
        "query_timestamp_ns",
        "realized_stop_index",
        "risk_digest",
        "scenario_library_digest",
        "scenario_set_digest",
        "schema_version",
        "selected_candidate_digest",
        "selected_candidate_index",
        "starting_equity",
        "state_snapshot_digest",
        "value_result_digest",
        "zero_candidate_index",
    }
    if set(manifest) != expected_fields:
        raise ValueError("C3 decision manifest field closure mismatch")
    if manifest["schema_version"] != C3_DECISION_ARTIFACT_SCHEMA:
        raise ValueError("unsupported C3 decision artifact schema")
    if manifest["decision_schema_version"] != C3_DECISION_SCHEMA:
        raise ValueError("unsupported persisted C3 decision schema")
    if manifest["arrays_file"] != _ARRAYS_NAME:
        raise ValueError("C3 decision arrays filename mismatch")

    arrays_payload = arrays_path.read_bytes()
    arrays_digest = _sha256_bytes(arrays_payload)
    if arrays_digest != _expect_string(
        manifest["arrays_digest"], field="arrays_digest"
    ):
        raise ValueError("C3 decision arrays digest mismatch")
    artifact_digest = require_sha256(
        _expect_string(manifest["artifact_digest"], field="artifact_digest"),
        field="artifact_digest",
    )
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("C3 decision artifact digest mismatch")
    if canonical_json_bytes(manifest) != manifest_path.read_bytes():
        raise ValueError("C3 decision manifest is not canonical JSON")

    metadata = _expect_mapping(manifest["array_metadata"], field="array_metadata")
    expected_arrays = {
        "projected_targets",
        "raw_candidate_actions",
        "regret",
        "score",
        "tie_candidate_indices",
    }
    if set(metadata) != expected_arrays:
        raise ValueError("C3 decision array metadata mismatch")
    try:
        with np.load(io.BytesIO(arrays_payload), allow_pickle=False) as loaded:
            if set(loaded.files) != expected_arrays:
                raise ValueError("C3 decision array closure mismatch")
            arrays = {name: np.asarray(loaded[name]) for name in expected_arrays}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and "closure" in str(error):
            raise
        raise ValueError("C3 decision arrays are invalid") from error
    if _deterministic_npz(arrays) != arrays_payload:
        raise ValueError("C3 decision arrays are not deterministic")
    for name, array in arrays.items():
        item = _expect_mapping(metadata[name], field=f"array_metadata.{name}")
        if set(item) != {"dtype", "shape"}:
            raise ValueError("C3 decision array metadata field closure mismatch")
        shape = item["shape"]
        if not isinstance(shape, list):
            raise ValueError("C3 decision array shape must be a list")
        if item["dtype"] != array.dtype.str or tuple(shape) != array.shape:
            raise ValueError("C3 decision array metadata does not match payload")

    candidate_raw = manifest["candidate_digests"]
    if not isinstance(candidate_raw, list):
        raise ValueError("candidate_digests must be a list")
    candidate_digests = tuple(
        _expect_string(value, field="candidate_digests") for value in candidate_raw
    )
    created = manifest["created_before_realized_replay"]
    if not isinstance(created, bool):
        raise ValueError("created_before_realized_replay must be boolean")
    decision = PersistedScenarioDecision(
        dataset_id=_expect_string(manifest["dataset_id"], field="dataset_id"),
        fold_digest=_expect_string(manifest["fold_digest"], field="fold_digest"),
        query_index=_expect_int(manifest["query_index"], field="query_index"),
        query_timestamp_ns=_expect_int(
            manifest["query_timestamp_ns"], field="query_timestamp_ns"
        ),
        state_snapshot_digest=_expect_string(
            manifest["state_snapshot_digest"], field="state_snapshot_digest"
        ),
        observation_digest=_expect_string(
            manifest["observation_digest"], field="observation_digest"
        ),
        environment_digest=_expect_string(
            manifest["environment_digest"], field="environment_digest"
        ),
        action_spec_digest=_expect_string(
            manifest["action_spec_digest"], field="action_spec_digest"
        ),
        execution_policy_digest=_expect_string(
            manifest["execution_policy_digest"], field="execution_policy_digest"
        ),
        risk_digest=_expect_string(manifest["risk_digest"], field="risk_digest"),
        starting_equity=_expect_number(
            manifest["starting_equity"], field="starting_equity"
        ),
        realized_stop_index=_expect_int(
            manifest["realized_stop_index"], field="realized_stop_index"
        ),
        scenario_library_digest=_expect_string(
            manifest["scenario_library_digest"], field="scenario_library_digest"
        ),
        scenario_set_digest=_expect_string(
            manifest["scenario_set_digest"], field="scenario_set_digest"
        ),
        candidate_generator_digest=_expect_string(
            manifest["candidate_generator_digest"],
            field="candidate_generator_digest",
        ),
        value_result_digest=_expect_string(
            manifest["value_result_digest"], field="value_result_digest"
        ),
        candidate_digests=candidate_digests,
        raw_candidate_actions=arrays["raw_candidate_actions"],
        projected_targets=arrays["projected_targets"],
        score=arrays["score"],
        regret=arrays["regret"],
        selected_candidate_index=_expect_int(
            manifest["selected_candidate_index"],
            field="selected_candidate_index",
        ),
        zero_candidate_index=_expect_int(
            manifest["zero_candidate_index"], field="zero_candidate_index"
        ),
        tie_candidate_indices=tuple(
            int(value) for value in arrays["tie_candidate_indices"].tolist()
        ),
        selected_candidate_digest=_expect_string(
            manifest["selected_candidate_digest"],
            field="selected_candidate_digest",
        ),
        created_before_realized_replay=created,
        decision_digest=_expect_string(
            manifest["decision_digest"], field="decision_digest"
        ),
    )
    return LoadedC3Decision(
        decision=decision,
        artifact_digest=artifact_digest,
        root=source,
    )
