"""Fail-closed staging of runtime-promotion evidence into training artifacts."""

from __future__ import annotations

from pathlib import Path

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_performance_io import (
    load_runtime_performance_evidence,
    write_runtime_performance_evidence,
)
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionReport,
    load_execution_promotion_report,
    write_execution_promotion_report,
)
from trade_rl.workflows.runtime_promotion_binding import (
    require_selection_execution_promotion,
)

RUNTIME_PROMOTION_REPORT_NAME = "runtime-promotion-report.json"
RUNTIME_PERFORMANCE_EVIDENCE_NAME = "runtime-performance-evidence.json"


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

    performance_evidence = None
    if report.evidence.performance_approved:
        performance_path = report_path.with_name(RUNTIME_PERFORMANCE_EVIDENCE_NAME)
        if not performance_path.is_file():
            raise FileNotFoundError(
                f"runtime performance evidence is missing: {performance_path}"
            )
        performance_evidence = load_runtime_performance_evidence(performance_path)
        if not performance_evidence.performance_approved:
            raise ValueError("runtime performance evidence is not approved")
        if performance_evidence.digest != report.performance_evidence_digest:
            raise ValueError("runtime performance evidence digest mismatch")

    if performance_evidence is not None:
        write_runtime_performance_evidence(
            stage / RUNTIME_PERFORMANCE_EVIDENCE_NAME,
            performance_evidence,
        )
    write_execution_promotion_report(stage / RUNTIME_PROMOTION_REPORT_NAME, report)
    return report


__all__ = [
    "RUNTIME_PERFORMANCE_EVIDENCE_NAME",
    "RUNTIME_PROMOTION_REPORT_NAME",
    "stage_training_runtime_promotion",
]
