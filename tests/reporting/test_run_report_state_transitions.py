from __future__ import annotations

from pathlib import Path

from tests.reporting.test_run_report_collector import (
    _write_admission_reject,
    _write_identity,
    _write_selection_pass,
    _write_signal_pass,
)
from trade_rl.reporting.run_report import RunStageStatus, build_run_report


def test_selection_terminal_evidence_requires_signal_pass(tmp_path: Path) -> None:
    config, _manifest = _write_identity(tmp_path)
    _write_selection_pass(tmp_path, config.candidates[0].digest)

    report = build_run_report(tmp_path)

    assert report.stages[0].status is RunStageStatus.MISSING
    assert report.stages[1].status is RunStageStatus.INVALID
    assert "upstream_not_passed_conflict" in report.stages[1].reasons


def test_admission_terminal_evidence_requires_selection_pass(tmp_path: Path) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    _write_admission_reject(
        tmp_path,
        selected_candidate_digest=config.candidates[0].digest,
    )

    report = build_run_report(tmp_path)

    assert report.stages[0].status is RunStageStatus.PASS
    assert report.stages[1].status is RunStageStatus.MISSING
    assert report.stages[2].status is RunStageStatus.INVALID
    assert "upstream_not_passed_conflict" in report.stages[2].reasons
