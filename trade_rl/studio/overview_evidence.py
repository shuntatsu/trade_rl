"""Compact Evidence status for the Studio overview decision cockpit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from trade_rl.studio.contracts import OverviewEvidenceSummary
from trade_rl.studio.evidence import inspect_run_evidence


def summarize_overview_evidence(
    root: Path | None,
    *,
    run_resource_id: str | None,
) -> OverviewEvidenceSummary:
    """Summarize required Evidence nodes without inventing timestamps."""
    if root is None or run_resource_id is None:
        return OverviewEvidenceSummary(
            run_resource_id=None,
            status="UNAVAILABLE",
            required_count=0,
            verified_count=0,
            blocker_count=0,
        )

    report = inspect_run_evidence(root, run_resource_id=run_resource_id)
    required = tuple(node for node in report.nodes if node.required)
    verified_count = sum(node.status == "VERIFIED" for node in required)
    blocker_count = sum(node.status != "VERIFIED" for node in required)
    manifest_blocked = any(
        node.key == "run_manifest" and node.status != "VERIFIED" for node in required
    )
    if report.files.status == "INVALID" and not manifest_blocked:
        blocker_count += 1

    status: Literal["VERIFIED", "INCOMPLETE", "INVALID"]
    if report.status == "INVALID" or blocker_count:
        status = "INVALID"
    elif verified_count == len(required):
        status = "VERIFIED"
    else:
        status = "INCOMPLETE"

    return OverviewEvidenceSummary(
        run_resource_id=run_resource_id,
        status=status,
        required_count=len(required),
        verified_count=verified_count,
        blocker_count=blocker_count,
    )


__all__ = ["summarize_overview_evidence"]
