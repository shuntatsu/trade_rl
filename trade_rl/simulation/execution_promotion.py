"""Fail-closed promotion evidence for conservative stateful execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_replay import (
    ExecutionEventArtifact,
    load_execution_event_artifact_bytes,
)

EXECUTION_EVIDENCE_FILE_NAME = "execution-evidence.json"
EXECUTION_EVIDENCE_SCHEMA = "execution_promotion_evidence_v3"
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
        before = path.lstat()
    except OSError as error:
        raise ExecutionPromotionError("execution event artifact is missing") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExecutionPromotionError(
            "execution event artifact must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow:
        flags |= no_follow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ExecutionPromotionError(
            "execution event artifact could not be opened safely"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ExecutionPromotionError(
                "execution event artifact must be a regular non-symlink file"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Dataset-, policy-, and event-artifact-bound promotion evidence."""

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
        artifact_fields = (
            self.order_event_artifact_digest,
            self.order_event_schema,
            self.terminal_book_digest,
            self.terminal_order_book_digest,
        )
        if any(value is not None for value in artifact_fields) or size > 0:
            if any(value is None for value in artifact_fields) or size <= 0:
                raise ValueError("execution event artifact identity is incomplete")
            assert self.order_event_artifact_digest is not None
            assert self.terminal_book_digest is not None
            assert self.terminal_order_book_digest is not None
            require_sha256(
                self.order_event_artifact_digest,
                field="order_event_artifact_digest",
            )
            require_sha256(self.terminal_book_digest, field="terminal_book_digest")
            require_sha256(
                self.terminal_order_book_digest,
                field="terminal_order_book_digest",
            )
            if not self.order_event_schema:
                raise ValueError("order_event_schema must be non-empty")
        if self.schema_version != EXECUTION_EVIDENCE_SCHEMA:
            raise ValueError("unsupported execution evidence schema")

    @property
    def digest(self) -> str:
        return content_digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "complete_order_evidence": self.complete_order_evidence,
            "dataset_id": self.dataset_id,
            "execution_policy_digest": self.execution_policy_digest,
            "order_event_artifact_digest": self.order_event_artifact_digest,
            "order_event_artifact_size_bytes": self.order_event_artifact_size_bytes,
            "order_event_count": self.order_event_count,
            "order_event_schema": self.order_event_schema,
            "partial_fill_carry": self.partial_fill_carry,
            "path_mode": self.path_mode,
            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,
            "schema_version": self.schema_version,
            "sensitivity_path_modes": self.sensitivity_path_modes,
            "terminal_book_digest": self.terminal_book_digest,
            "terminal_order_book_digest": self.terminal_order_book_digest,
            "trigger_volume_fractions": self.trigger_volume_fractions,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEvidence:
        required = {
            "complete_order_evidence",
            "dataset_id",
            "execution_policy_digest",
            "order_event_artifact_digest",
            "order_event_artifact_size_bytes",
            "order_event_count",
            "order_event_schema",
            "partial_fill_carry",
            "path_mode",
            "processing_bar_volume_capacity",
            "schema_version",
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
                value.get("order_event_schema"),
                field="order_event_schema",
            ),
            terminal_book_digest=_optional_string(
                value.get("terminal_book_digest"),
                field="terminal_book_digest",
            ),
            terminal_order_book_digest=_optional_string(
                value.get("terminal_order_book_digest"),
                field="terminal_order_book_digest",
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
    )


def validate_execution_event_artifact(
    evidence: ExecutionEvidence,
    event_path: Path,
) -> ExecutionEventArtifact:
    """Verify one exact canonical event artifact against signed evidence fields."""

    if (
        evidence.order_event_artifact_digest is None
        or evidence.order_event_artifact_size_bytes <= 0
        or evidence.order_event_schema is None
        or evidence.terminal_book_digest is None
        or evidence.terminal_order_book_digest is None
    ):
        raise ExecutionPromotionError(
            "execution promotion requires complete event artifact identity"
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
    return artifact


def validate_execution_promotion(
    evidence: ExecutionEvidence,
    *,
    expected_policy_digest: str,
    event_artifact_path: Path | None = None,
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
    validate_execution_event_artifact(evidence, event_artifact_path)
    return ExecutionPromotionDecision(
        promotable=True,
        evidence_digest=evidence.digest,
        execution_policy_digest=evidence.execution_policy_digest,
    )


def write_execution_evidence(path: Path, evidence: ExecutionEvidence) -> None:
    if path.exists():
        raise FileExistsError("execution evidence already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(evidence.to_mapping()) + b"\n")


def load_execution_evidence(path: Path) -> ExecutionEvidence:
    if not path.is_file():
        raise FileNotFoundError("execution evidence is missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("execution evidence must be a mapping")
    return ExecutionEvidence.from_mapping(raw)


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
