from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)
from trade_rl.simulation.runtime_performance_io import (
    load_runtime_performance_evidence,
    write_runtime_performance_evidence,
)
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
    write_execution_promotion_report,
)
from trade_rl.workflows.training_runtime_promotion import (
    RUNTIME_PERFORMANCE_EVIDENCE_NAME,
    stage_training_runtime_promotion,
)


def _performance_evidence() -> RuntimePerformanceEvidence:
    legacy = RuntimePerformanceMeasurement(
        timesteps=8,
        elapsed_seconds=4.0,
        steps_per_second=2.0,
        peak_self_rss_bytes=100,
        peak_children_rss_bytes=0,
        peak_process_tree_rss_bytes=100,
        peak_process_count=1,
    )
    nautilus = RuntimePerformanceMeasurement(
        timesteps=8,
        elapsed_seconds=4.0,
        steps_per_second=2.0,
        peak_self_rss_bytes=100,
        peak_children_rss_bytes=100,
        peak_process_tree_rss_bytes=200,
        peak_process_count=2,
    )
    return RuntimePerformanceEvidence(
        runtime_version="1.230.0",
        platform="linux-x86_64",
        algorithm="ppo",
        dataset_kind="deterministic_synthetic_btcusdt",
        source_digest="a" * 64,
        workloads=(
            RuntimePerformanceWorkload(
                timesteps=8,
                legacy_authoritative=legacy,
                nautilus_dual_shadow_streaming=nautilus,
            ),
        ),
        performance_approved=True,
        approval_policy_digest="b" * 64,
        approval_note="Reviewed test policy.",
    )


def _report(*, performance_evidence_digest: str):
    return build_execution_promotion_report(
        requested=RuntimeMode.NAUTILUS_AUTHORITATIVE,
        evidence=ExecutionPromotionEvidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=True,
            determinism_passed=True,
            performance_approved=True,
        ),
        representative_evidence_digest="8" * 64,
        performance_evidence_digest=performance_evidence_digest,
    )


def _proposal(*, report_digest: str) -> SelectionProposal:
    return SelectionProposal.create(
        walk_forward_run_digest="1" * 64,
        gate_evidence_digest="2" * 64,
        execution_sensitivity_digest="3" * 64,
        dataset_id="4" * 64,
        selected_configuration="candidate-a",
        candidate_config_digest="5" * 64,
        seeds=(7, 11),
        git_commit="a" * 40,
        dependency_digest="6" * 64,
        resume_checkpoint_digests=(),
        runtime_promotion_report_digest=report_digest,
    )


def test_matching_performance_artifact_is_staged_content_addressed(
    tmp_path: Path,
) -> None:
    performance = _performance_evidence()
    write_runtime_performance_evidence(
        tmp_path / RUNTIME_PERFORMANCE_EVIDENCE_NAME,
        performance,
    )
    report = _report(performance_evidence_digest=performance.digest)
    report_path = write_execution_promotion_report(
        tmp_path / "runtime-promotion-report.json",
        report,
    )
    stage = tmp_path / "stage"
    stage.mkdir()

    staged = stage_training_runtime_promotion(
        proposal=_proposal(report_digest=report.digest),
        report_path=report_path,
        stage=stage,
    )

    assert staged == report
    assert (
        load_runtime_performance_evidence(stage / RUNTIME_PERFORMANCE_EVIDENCE_NAME)
        == performance
    )


def test_performance_artifact_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    performance = _performance_evidence()
    write_runtime_performance_evidence(
        tmp_path / RUNTIME_PERFORMANCE_EVIDENCE_NAME,
        performance,
    )
    report = _report(performance_evidence_digest="c" * 64)
    report_path = write_execution_promotion_report(
        tmp_path / "runtime-promotion-report.json",
        report,
    )
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(ValueError, match="runtime performance evidence digest mismatch"):
        stage_training_runtime_promotion(
            proposal=_proposal(report_digest=report.digest),
            report_path=report_path,
            stage=stage,
        )


def test_unapproved_performance_artifact_cannot_back_approved_report(
    tmp_path: Path,
) -> None:
    performance = replace(
        _performance_evidence(),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observation only.",
    )
    write_runtime_performance_evidence(
        tmp_path / RUNTIME_PERFORMANCE_EVIDENCE_NAME,
        performance,
    )
    report = _report(performance_evidence_digest=performance.digest)
    report_path = write_execution_promotion_report(
        tmp_path / "runtime-promotion-report.json",
        report,
    )
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(ValueError, match="runtime performance evidence is not approved"):
        stage_training_runtime_promotion(
            proposal=_proposal(report_digest=report.digest),
            report_path=report_path,
            stage=stage,
        )
