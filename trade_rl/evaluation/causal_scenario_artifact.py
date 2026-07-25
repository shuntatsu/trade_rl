"""Deterministic artifacts for causal scenario action values."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_git_sha, require_sha256
from trade_rl.evaluation.causal_scenario_values import (
    CausalScenarioEvaluationResult,
    CausalScenarioEvaluatorConfig,
)

CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA: Final = "causal_scenario_value_artifact_v1"
CAUSAL_SCENARIO_MANIFEST_NAME: Final = "manifest.json"
CAUSAL_SCENARIO_ARRAYS_NAME: Final = "arrays.npz"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ALLOWED_FILES = frozenset({CAUSAL_SCENARIO_MANIFEST_NAME, CAUSAL_SCENARIO_ARRAYS_NAME})
_ARRAY_FIELDS: Final = (
    "raw_candidate_actions",
    "projected_targets",
    "scenario_probabilities",
    "scenario_anchor_indices",
    "scenario_distances",
    "query_condition",
    "anchor_conditions",
    "terminal_equity",
    "gross_log_returns",
    "baseline_relative_advantages",
    "filled_turnover",
    "interval_cost",
    "fill_ratio",
    "feasible_mask",
    "termination_codes",
    "mean_advantage",
    "loss_cvar",
    "score",
    "regret",
    "confidence_lower",
    "confidence_upper",
    "expected_filled_turnover",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
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
        raise FileNotFoundError(
            f"causal scenario artifact directory is missing: {root}"
        )
    actual: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("causal scenario artifact contains invalid file entry")
        actual.add(entry.name)
    if actual != _ALLOWED_FILES:
        raise ValueError("causal scenario artifact file closure mismatch")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{field} must be a sequence")


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    return tuple(
        _string(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )


def _integer_tuple(value: object, *, field: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )


def _config_from_payload(
    payload: Mapping[str, object],
) -> CausalScenarioEvaluatorConfig:
    expected = {
        "action_dimension",
        "bootstrap_resamples",
        "confidence_level",
        "cvar_alpha",
        "cvar_penalty",
        "horizon_decisions",
        "max_candidates",
        "probability_tolerance",
        "replay_tolerance",
        "scenario_count",
        "schema_version",
        "score_tolerance",
    }
    if set(payload) != expected:
        raise ValueError("causal scenario config payload field closure mismatch")
    return CausalScenarioEvaluatorConfig(
        action_dimension=_integer(
            payload["action_dimension"], field="config_payload.action_dimension"
        ),
        scenario_count=_integer(
            payload["scenario_count"], field="config_payload.scenario_count"
        ),
        horizon_decisions=_integer(
            payload["horizon_decisions"], field="config_payload.horizon_decisions"
        ),
        cvar_alpha=_finite_float(
            payload["cvar_alpha"], field="config_payload.cvar_alpha"
        ),
        cvar_penalty=_finite_float(
            payload["cvar_penalty"], field="config_payload.cvar_penalty"
        ),
        bootstrap_resamples=_integer(
            payload["bootstrap_resamples"],
            field="config_payload.bootstrap_resamples",
        ),
        confidence_level=_finite_float(
            payload["confidence_level"], field="config_payload.confidence_level"
        ),
        score_tolerance=_finite_float(
            payload["score_tolerance"], field="config_payload.score_tolerance"
        ),
        max_candidates=_integer(
            payload["max_candidates"], field="config_payload.max_candidates"
        ),
        replay_tolerance=_finite_float(
            payload["replay_tolerance"], field="config_payload.replay_tolerance"
        ),
        probability_tolerance=_finite_float(
            payload["probability_tolerance"],
            field="config_payload.probability_tolerance",
        ),
        schema_version=_string(
            payload["schema_version"], field="config_payload.schema_version"
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalScenarioValueArtifactManifest:
    artifact_digest: str
    arrays_digest: str
    result_digest: str
    query_digest: str
    config_digest: str
    scenario_set_digest: str
    scenario_ids: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    execution_intent_digests: tuple[str, ...]
    termination_reasons: tuple[str, ...]
    config_payload: dict[str, object]
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    candidate_generator_digest: str
    scenario_library_digest: str
    selected_candidate_index: int
    zero_candidate_index: int
    tie_candidate_indices: tuple[int, ...]
    array_metadata: dict[str, dict[str, object]]
    schema_version: str = CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "artifact_digest",
            "arrays_digest",
            "result_digest",
            "query_digest",
            "config_digest",
            "scenario_set_digest",
            "dataset_id",
            "fold_digest",
            "state_snapshot_digest",
            "observation_digest",
            "environment_digest",
            "action_spec_digest",
            "execution_policy_digest",
            "risk_digest",
            "trend_digest",
            "candidate_generator_digest",
            "scenario_library_digest",
        ):
            require_sha256(str(getattr(self, name)), field=name)
        require_git_sha(self.source_commit, field="source_commit")
        if self.schema_version != CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA:
            raise ValueError("unsupported causal scenario artifact schema")
        if set(self.array_metadata) != set(_ARRAY_FIELDS):
            raise ValueError("causal scenario array metadata mismatch")

    def base_payload(self) -> dict[str, object]:
        return {
            "action_spec_digest": self.action_spec_digest,
            "array_metadata": self.array_metadata,
            "arrays_digest": self.arrays_digest,
            "arrays_file": CAUSAL_SCENARIO_ARRAYS_NAME,
            "candidate_digests": self.candidate_digests,
            "candidate_generator_digest": self.candidate_generator_digest,
            "config_digest": self.config_digest,
            "config_payload": self.config_payload,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "execution_intent_digests": self.execution_intent_digests,
            "execution_policy_digest": self.execution_policy_digest,
            "fold_digest": self.fold_digest,
            "observation_digest": self.observation_digest,
            "query_digest": self.query_digest,
            "query_index": self.query_index,
            "query_timestamp_ns": self.query_timestamp_ns,
            "result_digest": self.result_digest,
            "risk_digest": self.risk_digest,
            "scenario_ids": self.scenario_ids,
            "scenario_library_digest": self.scenario_library_digest,
            "scenario_set_digest": self.scenario_set_digest,
            "schema_version": self.schema_version,
            "selected_candidate_index": self.selected_candidate_index,
            "source_commit": self.source_commit,
            "starting_equity": self.starting_equity,
            "state_snapshot_digest": self.state_snapshot_digest,
            "termination_reasons": self.termination_reasons,
            "tie_candidate_indices": self.tie_candidate_indices,
            "train_start": self.train_start,
            "train_stop": self.train_stop,
            "trend_digest": self.trend_digest,
            "zero_candidate_index": self.zero_candidate_index,
        }


def _arrays(result: CausalScenarioEvaluationResult) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(result, name)) for name in _ARRAY_FIELDS}


def _metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": np.asarray(value).dtype.str,
            "shape": tuple(int(size) for size in np.asarray(value).shape),
        }
        for name, value in sorted(arrays.items())
    }


def write_causal_scenario_value_artifact(
    root: str | Path, result: CausalScenarioEvaluationResult
) -> str:
    destination = Path(root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"causal scenario artifact destination is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(result)
    arrays_payload = _deterministic_npz(arrays)
    manifest = CausalScenarioValueArtifactManifest(
        artifact_digest="0" * 64,
        arrays_digest=_sha256_bytes(arrays_payload),
        result_digest=result.result_digest,
        query_digest=result.query_digest,
        config_digest=result.config.digest,
        scenario_set_digest=result.scenario_set_digest,
        scenario_ids=result.scenario_ids,
        candidate_digests=result.candidate_digests,
        execution_intent_digests=result.execution_intent_digests,
        termination_reasons=result.termination_reasons,
        config_payload=result.config.digest_payload(),
        dataset_id=result.dataset_id,
        fold_digest=result.fold_digest,
        train_start=result.train_start,
        train_stop=result.train_stop,
        query_index=result.query_index,
        query_timestamp_ns=result.query_timestamp_ns,
        source_commit=result.source_commit,
        state_snapshot_digest=result.state_snapshot_digest,
        observation_digest=result.observation_digest,
        environment_digest=result.environment_digest,
        action_spec_digest=result.action_spec_digest,
        execution_policy_digest=result.execution_policy_digest,
        risk_digest=result.risk_digest,
        trend_digest=result.trend_digest,
        starting_equity=result.starting_equity,
        candidate_generator_digest=result.candidate_generator_digest,
        scenario_library_digest=result.scenario_library_digest,
        selected_candidate_index=result.selected_candidate_index,
        zero_candidate_index=result.zero_candidate_index,
        tie_candidate_indices=result.tie_candidate_indices,
        array_metadata=_metadata(arrays),
    )
    base = manifest.base_payload()
    artifact_digest = content_digest(base)
    final = {**base, "artifact_digest": artifact_digest}
    _atomic_write(destination / CAUSAL_SCENARIO_ARRAYS_NAME, arrays_payload)
    _atomic_write(
        destination / CAUSAL_SCENARIO_MANIFEST_NAME, canonical_json_bytes(final)
    )
    return artifact_digest


def _load_manifest(root: Path) -> tuple[CausalScenarioValueArtifactManifest, bytes]:
    _verify_exact_files(root)
    raw = json.loads((root / CAUSAL_SCENARIO_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest_map = dict(_mapping(raw, field="manifest"))
    artifact_digest = _string(
        manifest_map.pop("artifact_digest", None), field="artifact_digest"
    )
    if content_digest(manifest_map) != artifact_digest:
        raise ValueError("causal scenario manifest digest mismatch")
    if manifest_map.pop("arrays_file", None) != CAUSAL_SCENARIO_ARRAYS_NAME:
        raise ValueError("causal scenario arrays file identity is invalid")
    expected_keys = {
        "action_spec_digest",
        "array_metadata",
        "arrays_digest",
        "candidate_digests",
        "candidate_generator_digest",
        "config_digest",
        "config_payload",
        "dataset_id",
        "environment_digest",
        "execution_intent_digests",
        "execution_policy_digest",
        "fold_digest",
        "observation_digest",
        "query_digest",
        "query_index",
        "query_timestamp_ns",
        "result_digest",
        "risk_digest",
        "scenario_ids",
        "scenario_library_digest",
        "scenario_set_digest",
        "schema_version",
        "selected_candidate_index",
        "source_commit",
        "starting_equity",
        "state_snapshot_digest",
        "termination_reasons",
        "tie_candidate_indices",
        "train_start",
        "train_stop",
        "trend_digest",
        "zero_candidate_index",
    }
    if set(manifest_map) != expected_keys:
        raise ValueError("causal scenario manifest field closure mismatch")
    manifest = CausalScenarioValueArtifactManifest(
        artifact_digest=artifact_digest,
        arrays_digest=_string(manifest_map["arrays_digest"], field="arrays_digest"),
        result_digest=_string(manifest_map["result_digest"], field="result_digest"),
        query_digest=_string(manifest_map["query_digest"], field="query_digest"),
        config_digest=_string(manifest_map["config_digest"], field="config_digest"),
        scenario_set_digest=_string(
            manifest_map["scenario_set_digest"], field="scenario_set_digest"
        ),
        scenario_ids=_string_tuple(manifest_map["scenario_ids"], field="scenario_ids"),
        candidate_digests=_string_tuple(
            manifest_map["candidate_digests"], field="candidate_digests"
        ),
        execution_intent_digests=_string_tuple(
            manifest_map["execution_intent_digests"],
            field="execution_intent_digests",
        ),
        termination_reasons=_string_tuple(
            manifest_map["termination_reasons"], field="termination_reasons"
        ),
        config_payload=dict(
            _mapping(manifest_map["config_payload"], field="config_payload")
        ),
        dataset_id=_string(manifest_map["dataset_id"], field="dataset_id"),
        fold_digest=_string(manifest_map["fold_digest"], field="fold_digest"),
        train_start=_integer(manifest_map["train_start"], field="train_start"),
        train_stop=_integer(manifest_map["train_stop"], field="train_stop"),
        query_index=_integer(manifest_map["query_index"], field="query_index"),
        query_timestamp_ns=_integer(
            manifest_map["query_timestamp_ns"], field="query_timestamp_ns"
        ),
        source_commit=_string(manifest_map["source_commit"], field="source_commit"),
        state_snapshot_digest=_string(
            manifest_map["state_snapshot_digest"], field="state_snapshot_digest"
        ),
        observation_digest=_string(
            manifest_map["observation_digest"], field="observation_digest"
        ),
        environment_digest=_string(
            manifest_map["environment_digest"], field="environment_digest"
        ),
        action_spec_digest=_string(
            manifest_map["action_spec_digest"], field="action_spec_digest"
        ),
        execution_policy_digest=_string(
            manifest_map["execution_policy_digest"], field="execution_policy_digest"
        ),
        risk_digest=_string(manifest_map["risk_digest"], field="risk_digest"),
        trend_digest=_string(manifest_map["trend_digest"], field="trend_digest"),
        starting_equity=_finite_float(
            manifest_map["starting_equity"], field="starting_equity"
        ),
        candidate_generator_digest=_string(
            manifest_map["candidate_generator_digest"],
            field="candidate_generator_digest",
        ),
        scenario_library_digest=_string(
            manifest_map["scenario_library_digest"], field="scenario_library_digest"
        ),
        selected_candidate_index=_integer(
            manifest_map["selected_candidate_index"], field="selected_candidate_index"
        ),
        zero_candidate_index=_integer(
            manifest_map["zero_candidate_index"], field="zero_candidate_index"
        ),
        tie_candidate_indices=_integer_tuple(
            manifest_map["tie_candidate_indices"], field="tie_candidate_indices"
        ),
        array_metadata={
            str(name): dict(_mapping(value, field=f"array_metadata.{name}"))
            for name, value in _mapping(
                manifest_map["array_metadata"], field="array_metadata"
            ).items()
        },
        schema_version=_string(manifest_map["schema_version"], field="schema_version"),
    )
    payload = (root / CAUSAL_SCENARIO_ARRAYS_NAME).read_bytes()
    if _sha256_bytes(payload) != manifest.arrays_digest:
        raise ValueError("causal scenario arrays digest mismatch")
    return manifest, payload


def load_causal_scenario_value_artifact(
    root: str | Path,
) -> CausalScenarioEvaluationResult:
    manifest, payload = _load_manifest(Path(root))
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAY_FIELDS):
            raise ValueError("causal scenario array names do not match manifest")
        for name in _ARRAY_FIELDS:
            meta = manifest.array_metadata[name]
            if set(meta) != {"dtype", "shape"}:
                raise ValueError(f"causal scenario array metadata invalid: {name}")
            array = np.asarray(archive[name])
            expected_shape = _integer_tuple(
                meta["shape"], field=f"array_metadata.{name}.shape"
            )
            expected_dtype = str(meta["dtype"])
            if array.shape != expected_shape:
                raise ValueError(f"causal scenario array shape mismatch: {name}")
            if array.dtype.str != expected_dtype:
                raise ValueError(f"causal scenario array dtype mismatch: {name}")
            arrays[name] = array
    config = _config_from_payload(manifest.config_payload)
    if config.digest != manifest.config_digest:
        raise ValueError("causal scenario config digest mismatch")
    result = CausalScenarioEvaluationResult(
        config=config,
        dataset_id=manifest.dataset_id,
        fold_digest=manifest.fold_digest,
        train_start=manifest.train_start,
        train_stop=manifest.train_stop,
        query_index=manifest.query_index,
        query_timestamp_ns=manifest.query_timestamp_ns,
        source_commit=manifest.source_commit,
        query_digest=manifest.query_digest,
        state_snapshot_digest=manifest.state_snapshot_digest,
        observation_digest=manifest.observation_digest,
        environment_digest=manifest.environment_digest,
        action_spec_digest=manifest.action_spec_digest,
        execution_policy_digest=manifest.execution_policy_digest,
        risk_digest=manifest.risk_digest,
        trend_digest=manifest.trend_digest,
        starting_equity=manifest.starting_equity,
        candidate_generator_digest=manifest.candidate_generator_digest,
        scenario_set_digest=manifest.scenario_set_digest,
        scenario_library_digest=manifest.scenario_library_digest,
        scenario_ids=manifest.scenario_ids,
        candidate_digests=manifest.candidate_digests,
        execution_intent_digests=manifest.execution_intent_digests,
        termination_reasons=manifest.termination_reasons,
        selected_candidate_index=manifest.selected_candidate_index,
        zero_candidate_index=manifest.zero_candidate_index,
        tie_candidate_indices=manifest.tie_candidate_indices,
        result_digest=manifest.result_digest,
        raw_candidate_actions=arrays["raw_candidate_actions"],
        projected_targets=arrays["projected_targets"],
        scenario_probabilities=arrays["scenario_probabilities"],
        scenario_anchor_indices=arrays["scenario_anchor_indices"],
        scenario_distances=arrays["scenario_distances"],
        query_condition=arrays["query_condition"],
        anchor_conditions=arrays["anchor_conditions"],
        terminal_equity=arrays["terminal_equity"],
        gross_log_returns=arrays["gross_log_returns"],
        baseline_relative_advantages=arrays["baseline_relative_advantages"],
        filled_turnover=arrays["filled_turnover"],
        interval_cost=arrays["interval_cost"],
        fill_ratio=arrays["fill_ratio"],
        feasible_mask=arrays["feasible_mask"],
        termination_codes=arrays["termination_codes"],
        mean_advantage=arrays["mean_advantage"],
        loss_cvar=arrays["loss_cvar"],
        score=arrays["score"],
        regret=arrays["regret"],
        confidence_lower=arrays["confidence_lower"],
        confidence_upper=arrays["confidence_upper"],
        expected_filled_turnover=arrays["expected_filled_turnover"],
    )
    return result


__all__ = [
    "CAUSAL_SCENARIO_ARRAYS_NAME",
    "CAUSAL_SCENARIO_MANIFEST_NAME",
    "CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA",
    "CausalScenarioValueArtifactManifest",
    "load_causal_scenario_value_artifact",
    "write_causal_scenario_value_artifact",
]
