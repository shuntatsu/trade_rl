"""Immutable runtime-promotion evidence handling for full research generations."""

from __future__ import annotations

from pathlib import Path

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    assess_runtime_performance,
)
from trade_rl.simulation.runtime_performance_io import (
    load_runtime_performance_evidence,
    load_runtime_performance_policy,
    write_runtime_performance_evidence,
    write_runtime_performance_policy,
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
RUNTIME_PERFORMANCE_POLICY_NAME = "runtime-performance-policy.json"


def _load_bound_performance_evidence(
    report: ExecutionPromotionReport,
    *,
    root: Path,
) -> RuntimePerformanceEvidence | None:
    if not report.evidence.performance_approved:
        return None
    path = root / RUNTIME_PERFORMANCE_EVIDENCE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"runtime performance evidence is missing: {path}")
    evidence = load_runtime_performance_evidence(path)
    if not evidence.performance_approved:
        raise ValueError("runtime performance evidence is not approved")
    if evidence.digest != report.performance_evidence_digest:
        raise ValueError("runtime performance evidence digest mismatch")
    return evidence


def _load_bound_performance_policy(
    evidence: RuntimePerformanceEvidence | None,
    *,
    root: Path,
) -> RuntimePerformanceApprovalPolicy | None:
    if evidence is None:
        return None
    path = root / RUNTIME_PERFORMANCE_POLICY_NAME
    if not path.is_file():
        raise FileNotFoundError(f"runtime performance policy is missing: {path}")
    policy = load_runtime_performance_policy(path)
    if evidence.approval_policy_digest != policy.digest:
        raise ValueError("runtime performance policy digest mismatch")
    decision = assess_runtime_performance(evidence=evidence, policy=policy)
    if not decision.approved:
        raise ValueError("runtime performance policy does not approve evidence")
    return policy


def retain_runtime_promotion_report(
    source: str | Path | None,
    *,
    work_root: Path,
) -> ExecutionPromotionReport | None:
    """Validate and retain one optional allowed promotion report immutably."""

    if source is None:
        return None
    source_path = Path(source)
    report = load_execution_promotion_report(source_path)
    if not report.decision.allowed:
        raise ValueError("runtime promotion report is not allowed")
    performance_evidence = _load_bound_performance_evidence(
        report,
        root=source_path.parent,
    )
    performance_policy = _load_bound_performance_policy(
        performance_evidence,
        root=source_path.parent,
    )
    if performance_evidence is not None:
        write_runtime_performance_evidence(
            work_root / RUNTIME_PERFORMANCE_EVIDENCE_NAME,
            performance_evidence,
        )
    if performance_policy is not None:
        write_runtime_performance_policy(
            work_root / RUNTIME_PERFORMANCE_POLICY_NAME,
            performance_policy,
        )
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
    performance_evidence = _load_bound_performance_evidence(report, root=work_root)
    _load_bound_performance_policy(performance_evidence, root=work_root)
    return report


__all__ = [
    "RUNTIME_PERFORMANCE_EVIDENCE_NAME",
    "RUNTIME_PERFORMANCE_POLICY_NAME",
    "RUNTIME_PROMOTION_REPORT_NAME",
    "require_retained_runtime_promotion",
    "retain_runtime_promotion_report",
]
