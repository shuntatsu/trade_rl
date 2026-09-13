"""Immutable authorization contract for sealed final evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments import (
    CANDIDATE_STRATEGY_NAMES,
    ContractViolationError,
)

FINAL_EVALUATION_AUTHORIZATION_SCHEMA = "final_evaluation_authorization_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolationError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if _SHA256_RE.fullmatch(text) is None:
        raise ContractViolationError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _canonical_ns_timestamp(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    try:
        instant = np.datetime64(text, "ns")
    except (TypeError, ValueError) as error:
        raise ContractViolationError(f"{field} must be a valid nanosecond timestamp") from error
    if np.isnat(instant):
        raise ContractViolationError(f"{field} must not be NaT")
    canonical = np.datetime_as_string(instant, unit="ns")
    if canonical != text:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        )
    return text


def _aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ContractViolationError(f"{field} must be timezone-aware")
    return value


def _datetime_from_payload(value: object, *, field: str) -> datetime:
    text = _text(value, field=field)
    normalized = text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ContractViolationError(f"{field} must be an ISO-8601 datetime") from error
    return _aware_datetime(parsed, field=field)


@dataclass(frozen=True, slots=True)
class FinalEvaluationAuthorization:
    """One-shot permission to open one preregistered unused-future window."""

    _PAYLOAD_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "study_digest",
            "study_freeze_digest",
            "winner_evidence_digest",
            "winner_strategy",
            "development_evaluation_stop_exclusive",
            "final_evaluation_start",
            "final_evaluation_stop_exclusive",
            "authorized_by",
            "authorized_at",
        }
    )

    study_digest: str
    study_freeze_digest: str
    winner_evidence_digest: str
    winner_strategy: str
    development_evaluation_stop_exclusive: str
    final_evaluation_start: str
    final_evaluation_stop_exclusive: str
    authorized_by: str
    authorized_at: datetime
    schema_version: str = FINAL_EVALUATION_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        study_digest = _sha256(self.study_digest, field="study_digest")
        study_freeze_digest = _sha256(
            self.study_freeze_digest,
            field="study_freeze_digest",
        )
        winner_evidence_digest = _sha256(
            self.winner_evidence_digest,
            field="winner_evidence_digest",
        )
        winner_strategy = _text(self.winner_strategy, field="winner_strategy")
        if winner_strategy not in CANDIDATE_STRATEGY_NAMES:
            raise ContractViolationError(
                "winner_strategy must be a registered candidate strategy"
            )
        development_stop = _canonical_ns_timestamp(
            self.development_evaluation_stop_exclusive,
            field="development_evaluation_stop_exclusive",
        )
        final_start = _canonical_ns_timestamp(
            self.final_evaluation_start,
            field="final_evaluation_start",
        )
        final_stop = _canonical_ns_timestamp(
            self.final_evaluation_stop_exclusive,
            field="final_evaluation_stop_exclusive",
        )
        if np.datetime64(final_start, "ns") < np.datetime64(development_stop, "ns"):
            raise ContractViolationError(
                "final evaluation start must not precede development evaluation stop"
            )
        if np.datetime64(final_stop, "ns") <= np.datetime64(final_start, "ns"):
            raise ContractViolationError(
                "final evaluation stop must be strictly later than final evaluation start"
            )
        authorized_by = _text(self.authorized_by, field="authorized_by")
        authorized_at = _aware_datetime(self.authorized_at, field="authorized_at")
        schema_version = _text(self.schema_version, field="schema_version")
        if schema_version != FINAL_EVALUATION_AUTHORIZATION_SCHEMA:
            raise ContractViolationError("unsupported final evaluation authorization schema")

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "study_freeze_digest", study_freeze_digest)
        object.__setattr__(self, "winner_evidence_digest", winner_evidence_digest)
        object.__setattr__(self, "winner_strategy", winner_strategy)
        object.__setattr__(
            self,
            "development_evaluation_stop_exclusive",
            development_stop,
        )
        object.__setattr__(self, "final_evaluation_start", final_start)
        object.__setattr__(self, "final_evaluation_stop_exclusive", final_stop)
        object.__setattr__(self, "authorized_by", authorized_by)
        object.__setattr__(self, "authorized_at", authorized_at)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "study_freeze_digest": self.study_freeze_digest,
            "winner_evidence_digest": self.winner_evidence_digest,
            "winner_strategy": self.winner_strategy,
            "development_evaluation_stop_exclusive": self.development_evaluation_stop_exclusive,
            "final_evaluation_start": self.final_evaluation_start,
            "final_evaluation_stop_exclusive": self.final_evaluation_stop_exclusive,
            "authorized_by": self.authorized_by,
            "authorized_at": self.authorized_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> FinalEvaluationAuthorization:
        if set(payload) != cls._PAYLOAD_KEYS:
            raise ContractViolationError("final authorization payload keys are malformed")
        return cls(
            study_digest=payload["study_digest"],
            study_freeze_digest=payload["study_freeze_digest"],
            winner_evidence_digest=payload["winner_evidence_digest"],
            winner_strategy=_text(payload["winner_strategy"], field="winner_strategy"),
            development_evaluation_stop_exclusive=_text(
                payload["development_evaluation_stop_exclusive"],
                field="development_evaluation_stop_exclusive",
            ),
            final_evaluation_start=_text(
                payload["final_evaluation_start"],
                field="final_evaluation_start",
            ),
            final_evaluation_stop_exclusive=_text(
                payload["final_evaluation_stop_exclusive"],
                field="final_evaluation_stop_exclusive",
            ),
            authorized_by=_text(payload["authorized_by"], field="authorized_by"),
            authorized_at=_datetime_from_payload(
                payload["authorized_at"],
                field="authorized_at",
            ),
            schema_version=_text(payload["schema_version"], field="schema_version"),
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = [
    "FINAL_EVALUATION_AUTHORIZATION_SCHEMA",
    "FinalEvaluationAuthorization",
]
