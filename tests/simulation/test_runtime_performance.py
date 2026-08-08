from __future__ import annotations

import pytest

from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
    assess_runtime_performance,
)


def _measurement(
    *,
    timesteps: int,
    elapsed_seconds: float,
    peak_self_rss_bytes: int,
    peak_children_rss_bytes: int,
    peak_process_tree_rss_bytes: int,
    peak_process_count: int,
) -> RuntimePerformanceMeasurement:
    return RuntimePerformanceMeasurement(
        timesteps=timesteps,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=timesteps / elapsed_seconds,
        peak_self_rss_bytes=peak_self_rss_bytes,
        peak_children_rss_bytes=peak_children_rss_bytes,
        peak_process_tree_rss_bytes=peak_process_tree_rss_bytes,
        peak_process_count=peak_process_count,
    )


def _workload(
    *,
    timesteps: int,
    legacy_elapsed: float,
    nautilus_elapsed: float,
    legacy_tree_rss: int,
    nautilus_tree_rss: int,
) -> RuntimePerformanceWorkload:
    return RuntimePerformanceWorkload(
        timesteps=timesteps,
        legacy_authoritative=_measurement(
            timesteps=timesteps,
            elapsed_seconds=legacy_elapsed,
            peak_self_rss_bytes=legacy_tree_rss,
            peak_children_rss_bytes=0,
            peak_process_tree_rss_bytes=legacy_tree_rss,
            peak_process_count=1,
        ),
        nautilus_dual_shadow_streaming=_measurement(
            timesteps=timesteps,
            elapsed_seconds=nautilus_elapsed,
            peak_self_rss_bytes=nautilus_tree_rss // 2,
            peak_children_rss_bytes=nautilus_tree_rss // 2,
            peak_process_tree_rss_bytes=nautilus_tree_rss,
            peak_process_count=2,
        ),
    )


def _evidence() -> RuntimePerformanceEvidence:
    return RuntimePerformanceEvidence(
        runtime_version="1.230.0",
        platform="linux-x86_64",
        algorithm="ppo",
        dataset_kind="deterministic_synthetic_btcusdt",
        workloads=(
            _workload(
                timesteps=8,
                legacy_elapsed=4.0,
                nautilus_elapsed=8.0,
                legacy_tree_rss=100,
                nautilus_tree_rss=180,
            ),
            _workload(
                timesteps=32,
                legacy_elapsed=8.0,
                nautilus_elapsed=24.0,
                legacy_tree_rss=120,
                nautilus_tree_rss=240,
            ),
        ),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observational evidence only.",
    )


def test_runtime_performance_evidence_reports_worst_ratios_and_stable_digest() -> None:
    evidence = _evidence()

    assert evidence.timesteps == (8, 32)
    assert evidence.worst_elapsed_slowdown_ratio == pytest.approx(3.0)
    assert evidence.worst_peak_process_tree_rss_ratio == pytest.approx(2.0)
    assert len(evidence.digest) == 64
    assert RuntimePerformanceEvidence.from_mapping(evidence.to_mapping()) == evidence


def test_runtime_performance_evidence_binds_workload_source_identity() -> None:
    evidence = RuntimePerformanceEvidence(
        runtime_version="1.230.0",
        platform="linux-x86_64",
        algorithm="ppo",
        dataset_kind="deterministic_synthetic_btcusdt",
        source_digest="a" * 64,
        workloads=(
            _workload(
                timesteps=8,
                legacy_elapsed=4.0,
                nautilus_elapsed=8.0,
                legacy_tree_rss=100,
                nautilus_tree_rss=180,
            ),
        ),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observational evidence only.",
    )

    mapping = evidence.to_mapping()
    assert mapping["source_digest"] == "a" * 64
    assert RuntimePerformanceEvidence.from_mapping(mapping) == evidence

    tampered = dict(mapping)
    tampered["source_digest"] = "b" * 64
    assert RuntimePerformanceEvidence.from_mapping(tampered).digest != evidence.digest


def test_runtime_performance_policy_must_be_reviewed_before_approval() -> None:
    evidence = _evidence()
    policy = RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=4.0,
        max_peak_process_tree_rss_ratio=2.5,
        minimum_workloads=2,
        minimum_max_timesteps=32,
        reviewed=False,
        review_reference=None,
    )

    decision = assess_runtime_performance(evidence=evidence, policy=policy)

    assert decision.approved is False
    assert decision.reasons == ("approval_policy_not_reviewed",)


def test_reviewed_runtime_performance_policy_uses_explicit_thresholds() -> None:
    evidence = _evidence()
    passing = RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=3.1,
        max_peak_process_tree_rss_ratio=2.1,
        minimum_workloads=2,
        minimum_max_timesteps=32,
        reviewed=True,
        review_reference="review-2026-08-09",
    )
    failing = RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=2.5,
        max_peak_process_tree_rss_ratio=1.9,
        minimum_workloads=2,
        minimum_max_timesteps=32,
        reviewed=True,
        review_reference="review-2026-08-09",
    )

    passing_decision = assess_runtime_performance(evidence=evidence, policy=passing)
    failing_decision = assess_runtime_performance(evidence=evidence, policy=failing)

    assert passing_decision.approved is True
    assert passing_decision.reasons == ()
    assert failing_decision.approved is False
    assert failing_decision.reasons == (
        "elapsed_slowdown_ratio_exceeded",
        "peak_process_tree_rss_ratio_exceeded",
    )
    assert passing.digest != failing.digest


def test_runtime_performance_measurement_rejects_inconsistent_memory_evidence() -> None:
    with pytest.raises(ValueError, match="process-tree RSS"):
        _measurement(
            timesteps=8,
            elapsed_seconds=4.0,
            peak_self_rss_bytes=200,
            peak_children_rss_bytes=0,
            peak_process_tree_rss_bytes=100,
            peak_process_count=1,
        )
