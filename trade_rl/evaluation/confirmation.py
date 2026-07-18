"""Public-key-authenticated fresh-confirmation evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_git_sha,
    require_sha256,
)
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    SignedEvidenceEnvelope,
    verify_signed_payload,
)

FRESH_CONFIRMATION_SCHEMA = "fresh_confirmation_evidence_ed25519_v4"
_CONFIRMATION_PURPOSE = "fresh-confirmation"
_DEFAULT_CLOCK_SKEW = timedelta(minutes=5)


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _returns(value: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if not result:
        raise ValueError("confirmation returns must not be empty")
    if any(not math.isfinite(item) or item < -1.0 for item in result):
        raise ValueError("confirmation returns must be finite and at least -1")
    return result


def _metrics(returns: tuple[float, ...]) -> tuple[float, float]:
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum_drawdown = max(maximum_drawdown, 1.0 - wealth / peak)
    return wealth - 1.0, maximum_drawdown


@dataclass(frozen=True, slots=True)
class FreshConfirmationEvidence:
    evidence_digest: str
    dataset_id: str
    environment_digest: str
    policy_digest: str
    training_run_digest: str
    git_commit: str
    dependency_digest: str
    required_after: datetime
    start_time: datetime
    end_time: datetime
    created_at: datetime
    returns: tuple[float, ...]
    return_period_hours: float
    order_log_digest: str
    fill_log_digest: str
    reconciliation_digest: str
    total_return: float
    maximum_drawdown: float
    days: float
    envelope: SignedEvidenceEnvelope
    sealed: bool = True
    schema_version: str = FRESH_CONFIRMATION_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_digest", self.evidence_digest),
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("policy_digest", self.policy_digest),
            ("training_run_digest", self.training_run_digest),
            ("dependency_digest", self.dependency_digest),
            ("order_log_digest", self.order_log_digest),
            ("fill_log_digest", self.fill_log_digest),
            ("reconciliation_digest", self.reconciliation_digest),
        ):
            require_sha256(value, field=name)
        require_git_sha(self.git_commit)
        for field in ("required_after", "start_time", "end_time", "created_at"):
            object.__setattr__(self, field, _utc(getattr(self, field), field=field))
        if self.end_time <= self.start_time:
            raise ValueError("confirmation end_time must follow start_time")
        object.__setattr__(self, "returns", _returns(self.returns))
        if self.sealed is not True:
            raise ValueError("confirmation evidence must be sealed")
        if self.schema_version != FRESH_CONFIRMATION_SCHEMA:
            raise ValueError("unsupported confirmation evidence schema")
        if (
            not math.isfinite(self.return_period_hours)
            or self.return_period_hours <= 0.0
        ):
            raise ValueError(
                "confirmation return_period_hours must be finite and positive"
            )
        total_return, maximum_drawdown = _metrics(self.returns)
        days = (self.end_time - self.start_time).total_seconds() / 86_400.0
        cadence_days = len(self.returns) * self.return_period_hours / 24.0
        if not math.isclose(days, cadence_days, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "confirmation return cadence does not cover the declared interval"
            )
        for name, observed, expected in (
            ("total_return", self.total_return, total_return),
            ("maximum_drawdown", self.maximum_drawdown, maximum_drawdown),
            ("days", self.days, days),
        ):
            if not math.isfinite(observed) or not math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"confirmation {name} does not match returns")
        if self.evidence_digest != content_digest(self.signed_payload()):
            raise ValueError("confirmation evidence digest mismatch")
        if self.evidence_digest != self.envelope.payload_digest:
            raise ValueError("confirmation envelope payload digest mismatch")
        if self.envelope.signed_at != self.created_at:
            raise ValueError("confirmation signed_at does not match created_at")
        if self.envelope.purpose != _CONFIRMATION_PURPOSE:
            raise ValueError("confirmation signature purpose mismatch")

    def signed_payload(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "days": self.days,
            "dependency_digest": self.dependency_digest,
            "end_time": self.end_time,
            "environment_digest": self.environment_digest,
            "fill_log_digest": self.fill_log_digest,
            "git_commit": self.git_commit,
            "maximum_drawdown": self.maximum_drawdown,
            "order_log_digest": self.order_log_digest,
            "policy_digest": self.policy_digest,
            "reconciliation_digest": self.reconciliation_digest,
            "required_after": self.required_after,
            "return_period_hours": self.return_period_hours,
            "returns": self.returns,
            "schema_version": self.schema_version,
            "sealed": self.sealed,
            "start_time": self.start_time,
            "total_return": self.total_return,
            "training_run_digest": self.training_run_digest,
        }

    def verify(
        self,
        trusted_keys: Mapping[str, PublicVerificationKey],
        *,
        expected_required_after: datetime,
        trusted_now: datetime,
        allowed_clock_skew: timedelta = _DEFAULT_CLOCK_SKEW,
    ) -> None:
        boundary = _utc(expected_required_after, field="expected_required_after")
        now = _utc(trusted_now, field="trusted_now")
        if allowed_clock_skew < timedelta(0):
            raise ValueError("allowed_clock_skew must not be negative")
        if self.required_after != boundary:
            raise ValueError("confirmation required boundary mismatch")
        if self.start_time < boundary:
            raise ValueError("confirmation start is not fresh after required boundary")
        if self.end_time > now + allowed_clock_skew:
            raise ValueError(
                "confirmation interval extends beyond trusted current time"
            )
        if self.created_at < self.end_time:
            raise ValueError("confirmation was created before collection completed")
        if self.created_at > now + allowed_clock_skew:
            raise ValueError("confirmation creation time is in the future")
        total_return, maximum_drawdown = _metrics(self.returns)
        if not math.isclose(total_return, self.total_return, abs_tol=1e-12):
            raise ValueError("confirmation return digest mismatch")
        if not math.isclose(maximum_drawdown, self.maximum_drawdown, abs_tol=1e-12):
            raise ValueError("confirmation drawdown digest mismatch")
        verify_signed_payload(
            self.signed_payload(),
            self.envelope,
            trusted_keys=trusted_keys,
            trusted_at=now,
            required_purpose=_CONFIRMATION_PURPOSE,
        )

    def with_returns(self, returns: Sequence[float]) -> FreshConfirmationEvidence:
        return replace(self, returns=tuple(float(item) for item in returns))


def write_confirmation_evidence(
    path: str | Path,
    evidence: FreshConfirmationEvidence,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(asdict(evidence))
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError(
                "refusing to overwrite immutable confirmation evidence"
            )
        return output
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return output


def load_confirmation_evidence(path: str | Path) -> FreshConfirmationEvidence:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("confirmation evidence must be an object")
    envelope_raw = raw.get("envelope")
    if not isinstance(envelope_raw, dict):
        raise ValueError("confirmation evidence envelope is missing")
    try:
        return FreshConfirmationEvidence(
            evidence_digest=_strict_string(raw, "evidence_digest"),
            dataset_id=_strict_string(raw, "dataset_id"),
            environment_digest=_strict_string(raw, "environment_digest"),
            policy_digest=_strict_string(raw, "policy_digest"),
            training_run_digest=_strict_string(raw, "training_run_digest"),
            git_commit=_strict_string(raw, "git_commit"),
            dependency_digest=_strict_string(raw, "dependency_digest"),
            required_after=_parse_datetime(raw, "required_after"),
            start_time=_parse_datetime(raw, "start_time"),
            end_time=_parse_datetime(raw, "end_time"),
            created_at=_parse_datetime(raw, "created_at"),
            returns=_strict_returns(raw["returns"]),
            return_period_hours=_strict_number(raw, "return_period_hours"),
            order_log_digest=_strict_string(raw, "order_log_digest"),
            fill_log_digest=_strict_string(raw, "fill_log_digest"),
            reconciliation_digest=_strict_string(raw, "reconciliation_digest"),
            total_return=_strict_number(raw, "total_return"),
            maximum_drawdown=_strict_number(raw, "maximum_drawdown"),
            days=_strict_number(raw, "days"),
            envelope=SignedEvidenceEnvelope.from_mapping(envelope_raw),
            sealed=_strict_bool(raw, "sealed"),
            schema_version=_strict_string(raw, "schema_version"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("confirmation evidence is invalid") from error


def _strict_string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _strict_number(raw: Mapping[str, object], field: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _strict_bool(raw: Mapping[str, object], field: str) -> bool:
    value = raw[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_returns(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise ValueError("returns must be a list")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw
    ):
        raise ValueError("returns must contain numbers")
    return tuple(float(item) for item in raw)


def _parse_datetime(raw: Mapping[str, object], field: str) -> datetime:
    value = _strict_string(raw, field)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error


__all__ = [
    "FRESH_CONFIRMATION_SCHEMA",
    "FreshConfirmationEvidence",
    "load_confirmation_evidence",
    "write_confirmation_evidence",
]
