"""Fail-closed resolution of named evaluation checks."""

from __future__ import annotations

from datetime import datetime

from trade_rl.domain.evaluation import GateCheck, GateDecision


def resolve_gate(
    checks: tuple[GateCheck, ...],
    *,
    decided_at: datetime,
) -> GateDecision:
    """Resolve a gate from mandatory checks without hidden policy defaults."""

    passed = all(check.passed for check in checks if check.mandatory)
    return GateDecision(
        passed=passed,
        checks=checks,
        decided_at=decided_at,
    )
