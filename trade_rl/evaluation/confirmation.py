"""Authenticated fresh-confirmation evidence with recomputed economic metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import (
    require_aware_datetime,
    require_git_sha,
    require_sha256,
)
from trade_rl.release.signing import (
    AuthenticatedEnvelope,
    sign_payload,
    verify_payload,
)

FRESH_CONFIRMATION_SCHEMA = "fresh_confirmation_evidence_v3"


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
    start_time: datetime
    end_time: datetime
    returns: tuple[float, ...]
    return_period_hours: float
    order_log_digest: str
    fill_log_digest: str
    reconciliation_digest: str
    total_return: float
    maximum_drawdown: float
    days: float
    envelope: AuthenticatedEnvelope
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
        require_aware_datetime(self.start_time, field="start_time")
        require_aware_datetime(self.end_time, field="end_time")
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
                observed, expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"confirmation {name} digest does not match returns")
        if self.evidence_digest != self.envelope.payload_digest:
            raise ValueError("confirmation evidence digest mismatch")

    def signed_payload(self) -> dict[str, object]:
        return {
            "training_run_digest": self.training_run_digest,
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
            "return_period_hours": self.return_period_hours,
            "returns": self.returns,
            "schema_version": self.schema_version,
            "sealed": self.sealed,
            "start_time": self.start_time,
            "total_return": self.total_return,
        }

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        environment_digest: str,
        policy_digest: str,
        training_run_digest: str,
        git_commit: str,
        dependency_digest: str,
        start_time: datetime,
        end_time: datetime,
        returns: Sequence[float],
        return_period_hours: float,
        order_log_digest: str,
        fill_log_digest: str,
        reconciliation_digest: str,
        key_id: str,
        signing_key: bytes | bytearray | memoryview,
    ) -> FreshConfirmationEvidence:
        resolved_returns = _returns(returns)
        total_return, maximum_drawdown = _metrics(resolved_returns)
        days = (end_time - start_time).total_seconds() / 86_400.0
        if not math.isfinite(return_period_hours) or return_period_hours <= 0.0:
            raise ValueError(
                "confirmation return_period_hours must be finite and positive"
            )
        cadence_days = len(resolved_returns) * return_period_hours / 24.0
        if not math.isclose(days, cadence_days, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "confirmation return cadence does not cover the declared interval"
            )
        payload = {
            "training_run_digest": training_run_digest,
            "dataset_id": dataset_id,
            "days": days,
            "dependency_digest": dependency_digest,
            "end_time": end_time,
            "environment_digest": environment_digest,
            "fill_log_digest": fill_log_digest,
            "git_commit": git_commit,
            "maximum_drawdown": maximum_drawdown,
            "order_log_digest": order_log_digest,
            "policy_digest": policy_digest,
            "reconciliation_digest": reconciliation_digest,
            "return_period_hours": return_period_hours,
            "returns": resolved_returns,
            "schema_version": FRESH_CONFIRMATION_SCHEMA,
            "sealed": True,
            "start_time": start_time,
            "total_return": total_return,
        }
        envelope = sign_payload(payload, key_id=key_id, signing_key=signing_key)
        return cls(
            evidence_digest=envelope.payload_digest,
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            policy_digest=policy_digest,
            training_run_digest=training_run_digest,
            git_commit=git_commit,
            dependency_digest=dependency_digest,
            start_time=start_time,
            end_time=end_time,
            returns=resolved_returns,
            return_period_hours=return_period_hours,
            order_log_digest=order_log_digest,
            fill_log_digest=fill_log_digest,
            reconciliation_digest=reconciliation_digest,
            total_return=total_return,
            maximum_drawdown=maximum_drawdown,
            days=days,
            envelope=envelope,
        )

    def verify(
        self,
        trusted_keys: Mapping[str, bytes | bytearray | memoryview],
    ) -> None:
        total_return, maximum_drawdown = _metrics(self.returns)
        if not math.isclose(total_return, self.total_return, abs_tol=1e-12):
            raise ValueError("confirmation return digest mismatch")
        if not math.isclose(maximum_drawdown, self.maximum_drawdown, abs_tol=1e-12):
            raise ValueError("confirmation drawdown digest mismatch")
        verify_payload(self.signed_payload(), self.envelope, trusted_keys=trusted_keys)

    def with_returns(self, returns: Sequence[float]) -> FreshConfirmationEvidence:
        return replace(self, returns=tuple(float(item) for item in returns))


def write_confirmation_evidence(
    path: str | Path,
    evidence: FreshConfirmationEvidence,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(asdict(evidence)))
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
            evidence_digest=str(raw["evidence_digest"]),
            dataset_id=str(raw["dataset_id"]),
            environment_digest=str(raw["environment_digest"]),
            policy_digest=str(raw["policy_digest"]),
            training_run_digest=str(raw["training_run_digest"]),
            git_commit=str(raw["git_commit"]),
            dependency_digest=str(raw["dependency_digest"]),
            start_time=datetime.fromisoformat(
                str(raw["start_time"]).replace("Z", "+00:00")
            ),
            end_time=datetime.fromisoformat(
                str(raw["end_time"]).replace("Z", "+00:00")
            ),
            returns=tuple(float(item) for item in raw["returns"]),
            return_period_hours=float(raw["return_period_hours"]),
            order_log_digest=str(raw["order_log_digest"]),
            fill_log_digest=str(raw["fill_log_digest"]),
            reconciliation_digest=str(raw["reconciliation_digest"]),
            total_return=float(raw["total_return"]),
            maximum_drawdown=float(raw["maximum_drawdown"]),
            days=float(raw["days"]),
            envelope=AuthenticatedEnvelope(
                key_id=str(envelope_raw["key_id"]),
                payload_digest=str(envelope_raw["payload_digest"]),
                signature=str(envelope_raw["signature"]),
                schema_version=str(envelope_raw["schema_version"]),
            ),
            sealed=bool(raw["sealed"]),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("confirmation evidence is invalid") from error


__all__ = [
    "FRESH_CONFIRMATION_SCHEMA",
    "FreshConfirmationEvidence",
    "load_confirmation_evidence",
    "write_confirmation_evidence",
]
