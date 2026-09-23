"""Immutable cross-Study evidence-consumption contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_sha256,
    contract_sha256_tuple,
    contract_text,
    contract_unique_enum_tuple,
)
from trade_rl.evaluation.experiments.errors import ContractViolationError


class EvidenceKind(StrEnum):
    """Durable identity class for development evidence consumed by later research."""

    EVIDENCE_SET = "EVIDENCE_SET"
    EXPERIMENT_DECISION = "EXPERIMENT_DECISION"
    STUDY_FREEZE = "STUDY_FREEZE"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"


class EvidenceUse(StrEnum):
    """How earlier development evidence influenced a later research definition."""

    HYPOTHESIS_FORMATION = "HYPOTHESIS_FORMATION"
    OBSERVATION_SELECTION = "OBSERVATION_SELECTION"
    MODEL_SELECTION = "MODEL_SELECTION"
    HYPERPARAMETER_SELECTION = "HYPERPARAMETER_SELECTION"
    EVALUATION_DESIGN = "EVALUATION_DESIGN"
    RESULT_INTERPRETATION = "RESULT_INTERPRETATION"


def _canonical_ns_timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        )
    try:
        instant = np.datetime64(value, "ns")
    except (TypeError, ValueError) as error:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        ) from error
    if np.isnat(instant):
        raise ContractViolationError(f"{field} must not be NaT")
    canonical = np.datetime_as_string(instant, unit="ns")
    if canonical != value:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        )
    return value


@dataclass(frozen=True, slots=True)
class ConsumedEvidence:
    """One development-evidence identity that informed a later Study."""

    evidence_kind: EvidenceKind
    evidence_digest: str
    development_start: str
    development_stop_exclusive: str
    uses: tuple[EvidenceUse, ...]
    schema_version: str = "consumed_research_evidence_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_kind, EvidenceKind):
            raise ContractViolationError("evidence_kind is unsupported")
        evidence_digest = contract_sha256(
            self.evidence_digest,
            field="evidence_digest",
        )
        start = _canonical_ns_timestamp(
            self.development_start,
            field="development_start",
        )
        stop = _canonical_ns_timestamp(
            self.development_stop_exclusive,
            field="development_stop_exclusive",
        )
        if np.datetime64(stop, "ns") <= np.datetime64(start, "ns"):
            raise ContractViolationError(
                "development_stop_exclusive must be strictly later than development_start"
            )
        uses = cast(
            tuple[EvidenceUse, ...],
            contract_unique_enum_tuple(
                self.uses,
                field="uses",
                expected_type=EvidenceUse,
            ),
        )
        uses = tuple(sorted(uses, key=lambda item: item.value))
        schema_version = contract_text(self.schema_version, field="schema_version")
        if schema_version != "consumed_research_evidence_v1":
            raise ContractViolationError("unsupported consumed evidence schema_version")

        object.__setattr__(self, "evidence_digest", evidence_digest)
        object.__setattr__(self, "development_start", start)
        object.__setattr__(self, "development_stop_exclusive", stop)
        object.__setattr__(self, "uses", uses)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind.value,
            "evidence_digest": self.evidence_digest,
            "development_start": self.development_start,
            "development_stop_exclusive": self.development_stop_exclusive,
            "uses": [item.value for item in self.uses],
        }

    @classmethod
    def from_payload(cls, value: object) -> ConsumedEvidence:
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise ContractViolationError("consumed evidence must be a JSON object")
        expected = {
            "schema_version",
            "evidence_kind",
            "evidence_digest",
            "development_start",
            "development_stop_exclusive",
            "uses",
        }
        if set(value) != expected:
            raise ContractViolationError("consumed evidence keys differ from contract")
        try:
            kind = EvidenceKind(value["evidence_kind"])
        except (TypeError, ValueError) as error:
            raise ContractViolationError("evidence_kind is unsupported") from error
        raw_uses = value["uses"]
        if not isinstance(raw_uses, list):
            raise ContractViolationError("uses must be an array")
        try:
            uses = tuple(EvidenceUse(item) for item in raw_uses)
        except (TypeError, ValueError) as error:
            raise ContractViolationError("uses contains an unsupported value") from error
        return cls(
            evidence_kind=kind,
            evidence_digest=value["evidence_digest"],  # type: ignore[arg-type]
            development_start=value["development_start"],  # type: ignore[arg-type]
            development_stop_exclusive=value["development_stop_exclusive"],  # type: ignore[arg-type]
            uses=uses,
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StudyResearchContext:
    """Canonical declaration of development evidence consumed before Study creation."""

    parent_context_digests: tuple[str, ...]
    consumed_evidence: tuple[ConsumedEvidence, ...]
    schema_version: str = "study_research_context_v1"

    def __post_init__(self) -> None:
        parents = contract_sha256_tuple(
            self.parent_context_digests,
            field="parent_context_digests",
        )
        parents = tuple(sorted(parents))
        if not isinstance(self.consumed_evidence, tuple):
            raise ContractViolationError("consumed_evidence must be a tuple")
        if any(
            not isinstance(item, ConsumedEvidence) for item in self.consumed_evidence
        ):
            raise ContractViolationError(
                "consumed_evidence contains an unsupported value"
            )
        consumed = tuple(
            sorted(self.consumed_evidence, key=lambda item: item.evidence_digest)
        )
        digests = tuple(item.evidence_digest for item in consumed)
        if len(set(digests)) != len(digests):
            raise ContractViolationError(
                "consumed_evidence must contain unique evidence digests"
            )
        schema_version = contract_text(self.schema_version, field="schema_version")
        if schema_version != "study_research_context_v1":
            raise ContractViolationError("unsupported research context schema_version")

        object.__setattr__(self, "parent_context_digests", parents)
        object.__setattr__(self, "consumed_evidence", consumed)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent_context_digests": list(self.parent_context_digests),
            "consumed_evidence": [item.to_payload() for item in self.consumed_evidence],
        }

    @classmethod
    def from_payload(cls, value: object) -> StudyResearchContext:
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise ContractViolationError("research_context must be a JSON object")
        expected = {
            "schema_version",
            "parent_context_digests",
            "consumed_evidence",
        }
        if set(value) != expected:
            raise ContractViolationError("research_context keys differ from contract")
        raw_parents = value["parent_context_digests"]
        raw_consumed = value["consumed_evidence"]
        if not isinstance(raw_parents, list):
            raise ContractViolationError("parent_context_digests must be an array")
        if not isinstance(raw_consumed, list):
            raise ContractViolationError("consumed_evidence must be an array")
        return cls(
            parent_context_digests=tuple(raw_parents),  # type: ignore[arg-type]
            consumed_evidence=tuple(
                ConsumedEvidence.from_payload(item) for item in raw_consumed
            ),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = [
    "ConsumedEvidence",
    "EvidenceKind",
    "EvidenceUse",
    "StudyResearchContext",
]
