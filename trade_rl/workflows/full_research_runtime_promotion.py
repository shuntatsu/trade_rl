"""Immutable runtime-promotion evidence handling for full research generations."""

from __future__ import annotations

from pathlib import Path

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionReport,
    load_execution_promotion_report,
    write_execution_promotion_report,
)
from trade_rl.workflows.runtime_promotion_binding import (
    require_selection_execution_promotion,
)

RUNTIME_PROMOTION_REPORT_NAME = "runtime-promotion-report.json"


def retain_runtime_promotion_report(
    source: Path | None,
    *,
    work_root: Path,
) -> ExecutionPromotionReport | None:
    """Validate and retain one optional allowed promotion report immutably."""

    if source is None:
        return None
    report = load_execution_promotion_report(source)
    if not report.decision.allowed:
        raise ValueError("runtime promotion report is not allowed")
    write_execution_promotion_report(
        work_root / RUNTIME_PROMOTION_REPORT_NAME,
        report,
    )
    return report


def require_retained_runtime_promotion(
    proposal: SelectionProposal,
    *,
    work_root: Path,
) -> ExecutionPromotionReport | None:
    """Fail closed unless retained evidence matches the signed proposal identity."""

    path = work_root / RUNTIME_PROMOTION_REPORT_NAME
    if proposal.runtime_promotion_report_digest is None:
        if path.exists():
            raise ValueError(
                "selection proposal does not authorize runtime promotion evidence"
            )
        return None
    if not path.is_file():
        raise FileNotFoundError(f"runtime promotion report is missing: {path}")
    report = load_execution_promotion_report(path)
    require_selection_execution_promotion(
        proposal=proposal,
        report=report,
        required_mode=report.requested_mode,
    )
    return report


__all__ = [
    "RUNTIME_PROMOTION_REPORT_NAME",
    "require_retained_runtime_promotion",
    "retain_runtime_promotion_report",
]
