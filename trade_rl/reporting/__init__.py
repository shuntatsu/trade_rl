"""Deterministic, read-only reporting for persisted research artifacts."""

from trade_rl.reporting.markdown import render_run_report_markdown
from trade_rl.reporting.run_report import (
    RUN_REPORT_STAGE_ORDER,
    RunReport,
    RunStageReport,
    RunStageStatus,
)

__all__ = [
    "RUN_REPORT_STAGE_ORDER",
    "RunReport",
    "RunStageReport",
    "RunStageStatus",
    "render_run_report_markdown",
]
