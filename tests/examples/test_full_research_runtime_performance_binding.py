from __future__ import annotations

import importlib
import sys
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

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"
PERFORMANCE_NAME = "runtime-performance-evidence.json"
REPORT_NAME = "runtime-promotion-report.json"


def _state_module():
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("run_full_research_state")


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


def _report(*, performance_digest: str):
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
        performance_evidence_digest=performance_digest,
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


def test_retain_authoritative_promotion_requires_performance_sidecar(
    tmp_path: Path,
) -> None:
    module = _state_module()
    performance = _performance_evidence()
    report = _report(performance_digest=performance.digest)
    source = write_execution_promotion_report(tmp_path / REPORT_NAME, report)
    work_root = tmp_path / "generation"
    work_root.mkdir()

    with pytest.raises(
        FileNotFoundError, match="runtime performance evidence is missing"
    ):
        module._retain_runtime_promotion_report(str(source), work_root=work_root)


def test_retain_authoritative_promotion_copies_bound_performance_sidecar(
    tmp_path: Path,
) -> None:
    module = _state_module()
    performance = _performance_evidence()
    write_runtime_performance_evidence(tmp_path / PERFORMANCE_NAME, performance)
    report = _report(performance_digest=performance.digest)
    source = write_execution_promotion_report(tmp_path / REPORT_NAME, report)
    work_root = tmp_path / "generation"
    work_root.mkdir()

    retained = module._retain_runtime_promotion_report(str(source), work_root=work_root)

    assert retained == report
    assert (
        load_runtime_performance_evidence(work_root / PERFORMANCE_NAME) == performance
    )


def test_finalize_recheck_requires_retained_performance_sidecar(tmp_path: Path) -> None:
    module = _state_module()
    performance = _performance_evidence()
    write_runtime_performance_evidence(tmp_path / PERFORMANCE_NAME, performance)
    report = _report(performance_digest=performance.digest)
    source = write_execution_promotion_report(tmp_path / REPORT_NAME, report)
    work_root = tmp_path / "generation"
    work_root.mkdir()
    module._retain_runtime_promotion_report(str(source), work_root=work_root)
    (work_root / PERFORMANCE_NAME).unlink()

    with pytest.raises(
        FileNotFoundError, match="runtime performance evidence is missing"
    ):
        module._require_retained_runtime_promotion(
            _proposal(report_digest=report.digest),
            work_root=work_root,
        )
