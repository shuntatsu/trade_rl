from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
    load_execution_promotion_report,
    write_execution_promotion_report,
)


def _stage_training_runtime_promotion():
    module = importlib.import_module("trade_rl.workflows.training_runtime_promotion")
    return module.stage_training_runtime_promotion


def _report():
    return build_execution_promotion_report(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=ExecutionPromotionEvidence(
            capability_passed=True,
            causal_bridge_passed=True,
            funding_passed=True,
            terminal_flat_passed=True,
            exact_parity_passed=False,
            determinism_passed=False,
            performance_approved=False,
        ),
    )


def _approved_performance_evidence() -> RuntimePerformanceEvidence:
    timesteps = 8
    legacy = RuntimePerformanceMeasurement(
        timesteps=timesteps,
        elapsed_seconds=4.0,
        steps_per_second=2.0,
        peak_self_rss_bytes=100,
        peak_children_rss_bytes=0,
        peak_process_tree_rss_bytes=100,
        peak_process_count=1,
    )
    nautilus = RuntimePerformanceMeasurement(
        timesteps=timesteps,
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
                timesteps=timesteps,
                legacy_authoritative=legacy,
                nautilus_dual_shadow_streaming=nautilus,
            ),
        ),
        performance_approved=True,
        approval_policy_digest="b" * 64,
        approval_note="Reviewed test policy.",
    )


def _authoritative_report(*, performance_evidence_digest: str):
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


def _proposal(*, runtime_digest: str | None) -> SelectionProposal:
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
        runtime_promotion_report_digest=runtime_digest,
    )


def test_legacy_training_requires_no_runtime_promotion_artifact(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    stage = tmp_path / "stage"
    stage.mkdir()

    assert (
        stage_training_runtime_promotion(
            proposal=None,
            report_path=None,
            stage=stage,
        )
        is None
    )
    assert not (stage / "runtime-promotion-report.json").exists()


def test_runtime_promotion_report_requires_selection_proposal(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    report = _report()
    report_path = write_execution_promotion_report(tmp_path / "report.json", report)
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        ValueError,
        match="runtime promotion report requires a selection proposal",
    ):
        stage_training_runtime_promotion(
            proposal=None,
            report_path=report_path,
            stage=stage,
        )


def test_unsigned_runtime_promotion_report_is_rejected(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    report = _report()
    report_path = write_execution_promotion_report(tmp_path / "report.json", report)
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        ValueError,
        match="selection proposal does not authorize runtime promotion evidence",
    ):
        stage_training_runtime_promotion(
            proposal=_proposal(runtime_digest=None),
            report_path=report_path,
            stage=stage,
        )


def test_runtime_bound_proposal_requires_report(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        ValueError,
        match="selected final training requires runtime promotion report",
    ):
        stage_training_runtime_promotion(
            proposal=_proposal(runtime_digest="f" * 64),
            report_path=None,
            stage=stage,
        )


def test_matching_runtime_promotion_is_staged_content_addressed(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    report = _report()
    report_path = write_execution_promotion_report(tmp_path / "report.json", report)
    stage = tmp_path / "stage"
    stage.mkdir()

    staged = stage_training_runtime_promotion(
        proposal=_proposal(runtime_digest=report.digest),
        report_path=report_path,
        stage=stage,
    )

    assert staged == report
    assert (
        load_execution_promotion_report(stage / "runtime-promotion-report.json")
        == report
    )


def test_mismatched_runtime_promotion_report_is_rejected(tmp_path: Path) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    report = _report()
    report_path = write_execution_promotion_report(tmp_path / "report.json", report)
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        ValueError,
        match="selection proposal runtime promotion report digest mismatch",
    ):
        stage_training_runtime_promotion(
            proposal=_proposal(runtime_digest="f" * 64),
            report_path=report_path,
            stage=stage,
        )


def test_authoritative_runtime_promotion_requires_persisted_performance_evidence(
    tmp_path: Path,
) -> None:
    stage_training_runtime_promotion = _stage_training_runtime_promotion()
    performance_evidence = _approved_performance_evidence()
    report = _authoritative_report(
        performance_evidence_digest=performance_evidence.digest,
    )
    report_path = write_execution_promotion_report(tmp_path / "report.json", report)
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(
        FileNotFoundError, match="runtime performance evidence is missing"
    ):
        stage_training_runtime_promotion(
            proposal=_proposal(runtime_digest=report.digest),
            report_path=report_path,
            stage=stage,
        )
