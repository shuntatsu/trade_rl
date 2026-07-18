"""Offline-only creation of fresh confirmation evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.evaluation.confirmation import (
    FRESH_CONFIRMATION_SCHEMA,
    FreshConfirmationEvidence,
    _metrics,
    _returns,
    _utc,
)
from trade_rl.release.offline_signing import sign_payload

_CONFIRMATION_PURPOSE = "fresh-confirmation"


def create_fresh_confirmation_evidence(
    *,
    dataset_id: str,
    environment_digest: str,
    policy_digest: str,
    training_run_digest: str,
    git_commit: str,
    dependency_digest: str,
    required_after: datetime,
    start_time: datetime,
    end_time: datetime,
    returns: Sequence[float],
    return_period_hours: float,
    order_log_digest: str,
    fill_log_digest: str,
    reconciliation_digest: str,
    created_at: datetime,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> FreshConfirmationEvidence:
    """Create and sign evidence after its complete collection interval."""

    resolved_returns = _returns(returns)
    total_return, maximum_drawdown = _metrics(resolved_returns)
    start = _utc(start_time, field="start_time")
    end = _utc(end_time, field="end_time")
    created = _utc(created_at, field="created_at")
    required = _utc(required_after, field="required_after")
    days = (end - start).total_seconds() / 86_400.0
    if not math.isfinite(return_period_hours) or return_period_hours <= 0.0:
        raise ValueError("confirmation return_period_hours must be finite and positive")
    cadence_days = len(resolved_returns) * return_period_hours / 24.0
    if not math.isclose(days, cadence_days, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("confirmation return cadence does not cover the declared interval")
    payload = {
        "created_at": created,
        "dataset_id": dataset_id,
        "days": days,
        "dependency_digest": dependency_digest,
        "end_time": end,
        "environment_digest": environment_digest,
        "fill_log_digest": fill_log_digest,
        "git_commit": git_commit,
        "maximum_drawdown": maximum_drawdown,
        "order_log_digest": order_log_digest,
        "policy_digest": policy_digest,
        "reconciliation_digest": reconciliation_digest,
        "required_after": required,
        "return_period_hours": return_period_hours,
        "returns": resolved_returns,
        "schema_version": FRESH_CONFIRMATION_SCHEMA,
        "sealed": True,
        "start_time": start,
        "total_return": total_return,
        "training_run_digest": training_run_digest,
    }
    envelope = sign_payload(
        payload,
        key_id=key_id,
        purpose=_CONFIRMATION_PURPOSE,
        private_key=private_key,
        signed_at=created,
    )
    return FreshConfirmationEvidence(
        evidence_digest=envelope.payload_digest,
        dataset_id=dataset_id,
        environment_digest=environment_digest,
        policy_digest=policy_digest,
        training_run_digest=training_run_digest,
        git_commit=git_commit,
        dependency_digest=dependency_digest,
        required_after=required,
        start_time=start,
        end_time=end,
        created_at=created,
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


__all__ = ["create_fresh_confirmation_evidence"]
