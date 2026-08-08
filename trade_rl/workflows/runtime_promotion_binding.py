"""Fail-closed binding between selected-final identity and execution promotion."""

from __future__ import annotations

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionReport,
    RuntimeMode,
)


def require_selection_execution_promotion(
    *,
    proposal: SelectionProposal,
    report: ExecutionPromotionReport,
    required_mode: RuntimeMode,
) -> None:
    """Require one exact allowed promotion report for selected-final execution."""

    if report.requested_mode != required_mode:
        raise ValueError("execution promotion mode mismatch")
    proposal.require_execution_evidence_digest(report.digest)
    if not report.decision.allowed:
        raise ValueError("execution promotion is not allowed")


__all__ = ["require_selection_execution_promotion"]
