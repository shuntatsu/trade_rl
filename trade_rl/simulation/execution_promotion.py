"""Fail-closed promotion evidence for conservative stateful execution."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.simulation.execution import ExecutionCostConfig

EXECUTION_EVIDENCE_FILE_NAME = "execution-evidence.json"
EXECUTION_EVIDENCE_SCHEMA = "execution_promotion_evidence_v1"
_DEFAULT_TRIGGER_VOLUME_FRACTIONS = (1.0, 0.5, 0.25, 0.0)
_PATH_MODES = frozenset({"optimistic", "neutral", "conservative"})


class ExecutionPromotionError(ValueError):
    """Raised when execution evidence cannot enter release promotion."""


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
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


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Dataset- and policy-bound evidence for execution-model promotion."""

    dataset_id: str
    execution_policy_digest: str
    path_mode: str
    processing_bar_volume_capacity: bool
    partial_fill_carry: bool
    trigger_volume_fractions: tuple[float, float, float, float]
    order_event_count: int
    complete_order_evidence: bool
    sensitivity_path_modes: tuple[str, ...] = ()
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
        if (
            isinstance(self.order_event_count, bool)
            or not isinstance(self.order_event_count, int)
            or self.order_event_count < 0
        ):
            raise ValueError("order_event_count must be a non-negative integer")
        if not isinstance(self.complete_order_evidence, bool):
            raise ValueError("complete_order_evidence must be a boolean")
        object.__setattr__(
            self,
            "sensitivity_path_modes",
            _path_modes(self.sensitivity_path_modes),
        )
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
            "order_event_count": self.order_event_count,
            "partial_fill_carry": self.partial_fill_carry,
            "path_mode": self.path_mode,
            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,
            "schema_version": self.schema_version,
            "sensitivity_path_modes": self.sensitivity_path_modes,
            "trigger_volume_fractions": self.trigger_volume_fractions,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExecutionEvidence:
        event_count = value.get("order_event_count")
        if isinstance(event_count, bool) or not isinstance(event_count, int):
            raise ValueError("order_event_count must be an integer")
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
            order_event_count=event_count,
            complete_order_evidence=_boolean(
                value.get("complete_order_evidence"),
                field="complete_order_evidence",
            ),
            sensitivity_path_modes=_path_modes(value.get("sensitivity_path_modes")),
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
) -> ExecutionEvidence:
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
    )


def validate_execution_promotion(
    evidence: ExecutionEvidence,
    *,
    expected_policy_digest: str,
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
    "validate_execution_promotion",
    "write_execution_evidence",
]
