"""Fail-closed staging of runtime-promotion evidence into training artifacts."""

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


def stage_training_runtime_promotion(
    *,
    proposal: SelectionProposal | None,
    report_path: Path | None,
    stage: Path,
) -> ExecutionPromotionReport | None:
    """Validate and immutably stage one selected-final runtime promotion report."""

    if proposal is None:
        if report_path is not None:
            raise ValueError("runtime promotion report requires a selection proposal")
        return None
    if proposal.runtime_promotion_report_digest is None:
        if report_path is not None:
            raise ValueError(
                "selection proposal does not authorize runtime promotion evidence"
            )
        return None
    if report_path is None:
        raise ValueError("selected final training requires runtime promotion report")
    report = load_execution_promotion_report(report_path)
    require_selection_execution_promotion(
        proposal=proposal,
        report=report,
        required_mode=report.requested_mode,
    )
    write_execution_promotion_report(stage / RUNTIME_PROMOTION_REPORT_NAME, report)
    return report


__all__ = [
    "RUNTIME_PROMOTION_REPORT_NAME",
    "stage_training_runtime_promotion",
]
