"""Fail-closed resolution of named evaluation checks."""

from __future__ import annotations

from datetime import datetime

from trade_rl.domain.evaluation import GateCheck, GateDecision


def resolve_gate(
    checks: tuple[GateCheck, ...],
    *,
    dataset_id: str,
    selected_policy_digest: str | None,
    evaluation_digest: str,
    decided_at: datetime,
) -> GateDecision:
    """Resolve a gate while preserving evaluated dataset and policy identity."""

    passed = all(check.passed for check in checks if check.mandatory)
    return GateDecision(
        dataset_id=dataset_id,
        selected_policy_digest=selected_policy_digest,
        evaluation_digest=evaluation_digest,
        passed=passed,
        checks=checks,
        decided_at=decided_at,
    )
