"""Fail-closed promotion evidence for conservative stateful execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_replay import (
    ExecutionEventArtifact,
    load_execution_event_artifact_bytes,
)

EXECUTION_EVIDENCE_FILE_NAME = "execution-evidence.json"
EXECUTION_EVIDENCE_SCHEMA = "execution_promotion_evidence_v4"
_DEFAULT_TRIGGER_VOLUME_FRACTIONS = (1.0, 0.5, 0.25, 0.0)
_PATH_MODES = frozenset({"optimistic", "neutral", "conservative"})


class ExecutionPromotionError(ValueError):
    """Raised when execution evidence cannot enter release promotion."""


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_non_negative_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value, field=field)


def _path_modes(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("sensitivity_path_modes must be a sequence")
    modes = tuple(_string(item, field="sensitivity_path_modes") for item in value)
    if any(mode not in _PATH_MODES for mode in modes):
        raise ValueError("sensitivity_path_modes contains an unsupported mode")
    if len(set(modes)) != len(modes):
        raise ValueError("sensitivity_path_modes must be unique")
    return modes


def _trigger_fractions(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("trigger_volume_fractions must be a sequence")
    fractions = tuple(float(item) for item in value)
    if len(fractions) != 4:
        raise ValueError("trigger_volume_fractions must contain four values")
    if not all(
        math.isfinite(item) and 0.0 <= item <= 1.0 for item in fractions
    ) or not all(fractions[index] >= fractions[index + 1] for index in range(3)):
        raise ValueError(
            "trigger_volume_fractions must be finite, bounded and non-increasing"
        )
    return (fractions[0], fractions[1], fractions[2], fractions[3])


def _regular_artifact_bytes(path: Path) -> bytes:
    try:
        with open_regular_binary(path, field="execution event artifact") as handle:
            return handle.read()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ExecutionPromotionError(
            "execution event artifact must be a readable regular non-symlink file"
        ) from error


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Dataset-, policy-, replay-, and event-artifact-bound promotion evidence."""

    dataset_id: str
    execution_policy_digest: str
    path_mode: str
    processing_bar_volume_capacity: bool
    partial_fill_carry: bool
    trigger_volume_fractions: tuple[float, float, float, float]
    order_event_count: int
    complete_order_evidence: bool
    sensitivity_path_modes: tuple[str, ...] = ()
    order_event_artifact_digest: str | None = None
    order_event_artifact_size_bytes: int = 0
    order_event_schema: str | None = None
    terminal_book_digest: str | None = None
    terminal_order_book_digest: str | None = None
    candidate_config_digest: str | None = None
    evaluation_run_digest: str | None = None
    fold: int | None = None
    seed: int | None = None
    replay_identity_digest: str | None = None
    replay_evidence_digest: str | None = None
    schema_version: str = EXECUTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="execution_evidence.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="execution_evidence.execution_policy_digest",
        )
        if self.path_mode not in _PATH_MODES:
            raise ValueError("execution evidence path_mode is unsupported")
        if not isinstance(self.processing_bar_volume_capacity, bool):
            raise ValueError("processing_bar_volume_capacity must be a boolean")
        if not isinstance(self.partial_fill_carry, bool):
            raise ValueError("partial_fill_carry must be a boolean")
        object.__setattr__(
            self,
            "trigger_volume_fractions",
            _trigger_fractions(self.trigger_volume_fractions),
        )
        _non_negative_integer(self.order_event_count, field="order_event_count")
        if not isinstance(self.complete_order_evidence, bool):
            raise ValueError("complete_order_evidence must be a boolean")
        object.__setattr__(
            self,
            "sensitivity_path_modes",
            _path_modes(self.sensitivity_path_modes),
        )
        size = _non_negative_integer(
            self.order_event_artifact_size_bytes,
            field="order_event_artifact_size_bytes",
        )
        fold = _optional_non_negative_integer(self.fold, field="fold")
        seed = _optional_non_negative_integer(self.seed, field="seed")
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "seed", seed)
        artifact_fields = (
            self.order_event_artifact_digest,
            self.order_event_schema,
            self.terminal_book_digest,
            self.terminal_order_book_digest,
            self.candidate_config_digest,
            self.evaluation_run_digest,
            fold,
            seed,
            self.replay_identity_digest,
            self.replay_evidence_digest,
        )
        if any(value is not None for value in artifact_fields) or size > 0:
            if any(value is None for value in artifact_fields) or size <= 0:
                raise ValueError("execution replay artifact identity is incomplete")
            for field, digest in (
                ("order_event_artifact_digest", self.order_event_artifact_digest),
                ("terminal_book_digest", self.terminal_book_digest),
                ("terminal_order_book_digest", self.terminal_order_book_digest),
                ("candidate_config_digest", self.candidate_config_digest),
                ("evaluation_run_digest", self.evaluation_run_digest),
                ("replay_identity_digest", self.replay_identity_digest),
                ("replay_evidence_digest", self.replay_evidence_digest),
            ):
                assert digest is not None
                require_sha256(digest, field=field)
            if not self.order_event_schema:
                raise ValueError("order_event_schema must be non-empty")
        if self.schema_version != EXECUTION_EVIDENCE_SCHEMA:
            raise ValueError("unsupported execution evidence schema")

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "complete_order_evidence": self.complete_order_evidence,
            "dataset_id": self.dataset_id,
            "evaluation_run_digest": self.evaluation_run_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "fold": self.fold,
            "order_event_artifact_digest": self.order_event_artifact_digest,
            "order_event_artifact_size_bytes": self.order_event_artifact_size_bytes,
            "order_event_count": self.order_event_count,
            "order_event_schema": self.order_event_schema,
            "partial_fill_carry": self.partial_fill_carry,
            "path_mode": self.path_mode,
            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,
            "replay_evidence_digest": self.replay_evidence_digest,
            "replay_identity_digest": self.replay_identity_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "sensitivity_path_modes": self.sensitivity_path_modes,
            "terminal_book_digest": self.terminal_book_digest,
            "terminal_order_book_digest": self.terminal_order_book_digest,
            "trigger_volume_fractions": self.trigger_volume_fractions,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEvidence:
        required = {
            "candidate_config_digest",
            "complete_order_evidence",
            "dataset_id",
            "evaluation_run_digest",
            "execution_policy_digest",
            "fold",
            "order_event_artifact_digest",
            "order_event_artifact_size_bytes",
            "order_event_count",
            "order_event_schema",
            "partial_fill_carry",
            "path_mode",
            "processing_bar_volume_capacity",
            "replay_evidence_digest",
            "replay_identity_digest",
            "schema_version",
            "seed",
            "sensitivity_path_modes",
            "terminal_book_digest",
            "terminal_order_book_digest",
            "trigger_volume_fractions",
        }
        if set(value) != required:
            raise ValueError("execution evidence field closure mismatch")
        return cls(
            dataset_id=_string(value.get("dataset_id"), field="dataset_id"),
            execution_policy_digest=_string(
                value.get("execution_policy_digest"),
                field="execution_policy_digest",
            ),
            path_mode=_string(value.get("path_mode"), field="path_mode"),
            processing_bar_volume_capacity=_boolean(
                value.get("processing_bar_volume_capacity"),
                field="processing_bar_volume_capacity",
            ),
            partial_fill_carry=_boolean(
                value.get("partial_fill_carry"),
                field="partial_fill_carry",
            ),
            trigger_volume_fractions=_trigger_fractions(
                value.get("trigger_volume_fractions")
            ),
            order_event_count=_non_negative_integer(
                value.get("order_event_count"),
                field="order_event_count",
            ),
            complete_order_evidence=_boolean(
                value.get("complete_order_evidence"),
                field="complete_order_evidence",
            ),
            sensitivity_path_modes=_path_modes(value.get("sensitivity_path_modes")),
            order_event_artifact_digest=_optional_string(
                value.get("order_event_artifact_digest"),
                field="order_event_artifact_digest",
            ),
            order_event_artifact_size_bytes=_non_negative_integer(
                value.get("order_event_artifact_size_bytes"),
                field="order_event_artifact_size_bytes",
            ),
            order_event_schema=_optional_string(
                value.get("order_event_schema"), field="order_event_schema"
            ),
            terminal_book_digest=_optional_string(
                value.get("terminal_book_digest"), field="terminal_book_digest"
            ),
            terminal_order_book_digest=_optional_string(
                value.get("terminal_order_book_digest"),
                field="terminal_order_book_digest",
            ),
            candidate_config_digest=_optional_string(
                value.get("candidate_config_digest"),
                field="candidate_config_digest",
            ),
            evaluation_run_digest=_optional_string(
                value.get("evaluation_run_digest"), field="evaluation_run_digest"
            ),
            fold=_optional_non_negative_integer(value.get("fold"), field="fold"),
            seed=_optional_non_negative_integer(value.get("seed"), field="seed"),
            replay_identity_digest=_optional_string(
                value.get("replay_identity_digest"),
                field="replay_identity_digest",
            ),
            replay_evidence_digest=_optional_string(
                value.get("replay_evidence_digest"),
                field="replay_evidence_digest",
            ),
            schema_version=_string(value.get("schema_version"), field="schema_version"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPromotionDecision:
    promotable: bool
    evidence_digest: str
    execution_policy_digest: str


def execution_evidence_from_cost(
    *,
    dataset_id: str,
    cost: ExecutionCostConfig,
    order_event_count: int = 0,
    complete_order_evidence: bool = False,
    sensitivity_path_modes: tuple[str, ...] = (),
    order_event_artifact_path: Path | None = None,
) -> ExecutionEvidence:
    artifact_digest: str | None = None
    artifact_size = 0
    order_event_schema: str | None = None
    terminal_book_digest: str | None = None
    terminal_order_book_digest: str | None = None
    candidate_config_digest: str | None = None
    evaluation_run_digest: str | None = None
    fold: int | None = None
    seed: int | None = None
    replay_identity_digest: str | None = None
    replay_evidence_digest: str | None = None
    if order_event_artifact_path is not None:
        if order_event_count != 0 or complete_order_evidence:
            raise ValueError(
                "event-derived evidence may not accept claimed event count or completeness"
            )
        raw = _regular_artifact_bytes(order_event_artifact_path)
        artifact = load_execution_event_artifact_bytes(raw)
        if artifact.dataset_id != dataset_id:
            raise ValueError("execution event artifact dataset identity mismatch")
        if artifact.execution_policy_digest != cost.execution_policy_digest:
            raise ValueError("execution event artifact policy identity mismatch")
        artifact_digest = hashlib.sha256(raw).hexdigest()
        artifact_size = len(raw)
        order_event_count = artifact.order_event_count
        complete_order_evidence = True
        order_event_schema = artifact.order_event_schema
        terminal_book_digest = artifact.terminal_book_digest
        terminal_order_book_digest = artifact.terminal_order_book_digest
        candidate_config_digest = artifact.replay_identity.candidate_config_digest
        evaluation_run_digest = artifact.replay_identity.evaluation_run_digest
        fold = artifact.replay_identity.fold
        seed = artifact.replay_identity.seed
        replay_identity_digest = artifact.replay_identity.digest
        replay_evidence_digest = artifact.replay_evidence.digest
    return ExecutionEvidence(
        dataset_id=dataset_id,
        execution_policy_digest=cost.execution_policy_digest,
        path_mode=cost.path_mode,
        processing_bar_volume_capacity=cost.processing_bar_volume_capacity,
        partial_fill_carry=cost.partial_fill_carry,
        trigger_volume_fractions=cost.trigger_volume_fractions,
        order_event_count=order_event_count,
        complete_order_evidence=complete_order_evidence,
        sensitivity_path_modes=sensitivity_path_modes,
        order_event_artifact_digest=artifact_digest,
        order_event_artifact_size_bytes=artifact_size,
        order_event_schema=order_event_schema,
        terminal_book_digest=terminal_book_digest,
        terminal_order_book_digest=terminal_order_book_digest,
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=evaluation_run_digest,
        fold=fold,
        seed=seed,
        replay_identity_digest=replay_identity_digest,
        replay_evidence_digest=replay_evidence_digest,
    )


def validate_execution_event_artifact(
    evidence: ExecutionEvidence,
    event_path: Path,
) -> ExecutionEventArtifact:
    """Verify one exact canonical replay artifact against signed evidence fields."""

    required = (
        evidence.order_event_artifact_digest,
        evidence.order_event_schema,
        evidence.terminal_book_digest,
        evidence.terminal_order_book_digest,
        evidence.candidate_config_digest,
        evidence.evaluation_run_digest,
        evidence.fold,
        evidence.seed,
        evidence.replay_identity_digest,
        evidence.replay_evidence_digest,
    )
    if any(value is None for value in required) or (
        evidence.order_event_artifact_size_bytes <= 0
    ):
        raise ExecutionPromotionError(
            "execution promotion requires complete replay artifact identity"
        )
    raw = _regular_artifact_bytes(event_path)
    if len(raw) != evidence.order_event_artifact_size_bytes:
        raise ExecutionPromotionError("execution event artifact size mismatch")
    if hashlib.sha256(raw).hexdigest() != evidence.order_event_artifact_digest:
        raise ExecutionPromotionError("execution event artifact digest mismatch")
    try:
        artifact = load_execution_event_artifact_bytes(raw)
    except ValueError as error:
        raise ExecutionPromotionError("execution event artifact is invalid") from error
    if artifact.dataset_id != evidence.dataset_id:
        raise ExecutionPromotionError("execution event artifact dataset mismatch")
    if artifact.execution_policy_digest != evidence.execution_policy_digest:
        raise ExecutionPromotionError("execution event artifact policy mismatch")
    if artifact.order_event_schema != evidence.order_event_schema:
        raise ExecutionPromotionError("execution event schema mismatch")
    if artifact.order_event_count != evidence.order_event_count:
        raise ExecutionPromotionError("execution event count mismatch")
    if artifact.terminal_book_digest != evidence.terminal_book_digest:
        raise ExecutionPromotionError("execution terminal book digest mismatch")
    if artifact.terminal_order_book_digest != evidence.terminal_order_book_digest:
        raise ExecutionPromotionError("execution terminal order book digest mismatch")
    if artifact.replay_identity.candidate_config_digest != evidence.candidate_config_digest:
        raise ExecutionPromotionError("execution candidate identity mismatch")
    if artifact.replay_identity.evaluation_run_digest != evidence.evaluation_run_digest:
        raise ExecutionPromotionError("execution evaluation run identity mismatch")
    if artifact.replay_identity.fold != evidence.fold:
        raise ExecutionPromotionError("execution fold identity mismatch")
    if artifact.replay_identity.seed != evidence.seed:
        raise ExecutionPromotionError("execution seed identity mismatch")
    if artifact.replay_identity.digest != evidence.replay_identity_digest:
        raise ExecutionPromotionError("execution replay identity digest mismatch")
    if artifact.replay_evidence.digest != evidence.replay_evidence_digest:
        raise ExecutionPromotionError("execution replay evidence digest mismatch")
    return artifact


def validate_execution_promotion(
    evidence: ExecutionEvidence,
    *,
    expected_policy_digest: str,
    event_artifact_path: Path | None = None,
    expected_candidate_config_digest: str | None = None,
    expected_evaluation_run_digest: str | None = None,
    expected_fold: int | None = None,
    expected_seed: int | None = None,
) -> ExecutionPromotionDecision:
    require_sha256(expected_policy_digest, field="expected_policy_digest")
    if evidence.execution_policy_digest != expected_policy_digest:
        raise ExecutionPromotionError("execution policy digest mismatch")
    if evidence.path_mode != "conservative":
        raise ExecutionPromotionError(
            "execution promotion requires conservative primary evidence"
        )
    if not evidence.processing_bar_volume_capacity:
        raise ExecutionPromotionError(
            "execution promotion requires processing-bar volume capacity"
        )
    if not evidence.partial_fill_carry:
        raise ExecutionPromotionError("execution promotion requires partial-fill carry")
    if evidence.order_event_count <= 0:
        raise ExecutionPromotionError(
            "execution promotion requires at least one order event"
        )
    if not evidence.complete_order_evidence:
        raise ExecutionPromotionError(
            "execution promotion requires complete order evidence"
        )
    if evidence.sensitivity_path_modes and (
        "conservative" not in evidence.sensitivity_path_modes
    ):
        raise ExecutionPromotionError(
            "execution sensitivity evidence must include conservative mode"
        )
    if any(
        actual > maximum + 1e-12
        for actual, maximum in zip(
            evidence.trigger_volume_fractions,
            _DEFAULT_TRIGGER_VOLUME_FRACTIONS,
            strict=True,
        )
    ):
        raise ExecutionPromotionError(
            "execution trigger volume fractions are less conservative than required"
        )
    if event_artifact_path is None:
        raise ExecutionPromotionError(
            "execution promotion requires the bound event artifact"
        )
    artifact = validate_execution_event_artifact(evidence, event_artifact_path)
    for expected, actual, field in (
        (
            expected_candidate_config_digest,
            artifact.replay_identity.candidate_config_digest,
            "candidate",
        ),
        (
            expected_evaluation_run_digest,
            artifact.replay_identity.evaluation_run_digest,
            "evaluation run",
        ),
    ):
        if expected is not None:
            require_sha256(expected, field=f"expected_{field.replace(' ', '_')}")
            if expected != actual:
                raise ExecutionPromotionError(f"execution {field} identity mismatch")
    for expected, actual, field in (
        (expected_fold, artifact.replay_identity.fold, "fold"),
        (expected_seed, artifact.replay_identity.seed, "seed"),
    ):
        if expected is not None:
            _non_negative_integer(expected, field=f"expected_{field}")
            if expected != actual:
                raise ExecutionPromotionError(f"execution {field} identity mismatch")
    return ExecutionPromotionDecision(
        promotable=True,
        evidence_digest=evidence.digest,
        execution_policy_digest=evidence.execution_policy_digest,
    )


def write_execution_evidence(path: Path, evidence: ExecutionEvidence) -> None:
    payload = canonical_json_bytes(evidence.to_mapping()) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("execution evidence already exists") from None
    _fsync_directory(path.parent)


def load_execution_evidence(path: Path) -> ExecutionEvidence:
    try:
        with open_regular_binary(path, field="execution evidence") as handle:
            raw_bytes = handle.read()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise FileNotFoundError("execution evidence is missing or unsafe") from error
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution evidence must be valid JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError("execution evidence must be a mapping")
    evidence = ExecutionEvidence.from_mapping(raw)
    if raw_bytes != canonical_json_bytes(evidence.to_mapping()) + b"\n":
        raise ValueError("execution evidence must use canonical encoding")
    return evidence


__all__ = [
    "EXECUTION_EVIDENCE_FILE_NAME",
    "EXECUTION_EVIDENCE_SCHEMA",
    "ExecutionEvidence",
    "ExecutionPromotionDecision",
    "ExecutionPromotionError",
    "execution_evidence_from_cost",
    "load_execution_evidence",
    "validate_execution_event_artifact",
    "validate_execution_promotion",
    "write_execution_evidence",
]
