"""Fail-closed resolution of evidence-bound evaluation checks."""

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
    require_evidence: bool = False,
) -> GateDecision:
    """Resolve a gate and optionally require recomputable metric evidence."""

    if require_evidence and any(
        check.mandatory and not check.evidence_bound for check in checks
    ):
        raise ValueError("release-eligible mandatory checks require metric evidence")
    passed = all(check.passed for check in checks if check.mandatory)
    return GateDecision(
        dataset_id=dataset_id,
        selected_policy_digest=selected_policy_digest,
        evaluation_digest=evaluation_digest,
        passed=passed,
        checks=checks,
        decided_at=decided_at,
    )
