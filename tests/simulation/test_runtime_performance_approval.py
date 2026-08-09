from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

import trade_rl.simulation.runtime_performance as runtime_performance
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)


def _measurement(*, timesteps: int, elapsed_seconds: float, rss: int) -> RuntimePerformanceMeasurement:
    return RuntimePerformanceMeasurement(
        timesteps=timesteps,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=timesteps / elapsed_seconds,
        peak_self_rss_bytes=rss,
        peak_children_rss_bytes=0,
        peak_process_tree_rss_bytes=rss,
        peak_process_count=1,
    )


def _observational_evidence() -> RuntimePerformanceEvidence:
    return RuntimePerformanceEvidence(
        runtime_version="1.230.0",
        platform="linux-x86_64",
        algorithm="ppo",
        dataset_kind="deterministic_synthetic_btcusdt",
        source_digest="a" * 64,
        workloads=(
            RuntimePerformanceWorkload(
                timesteps=32,
                legacy_authoritative=_measurement(
                    timesteps=32,
                    elapsed_seconds=8.0,
                    rss=100,
                ),
                nautilus_dual_shadow_streaming=_measurement(
                    timesteps=32,
                    elapsed_seconds=24.0,
                    rss=200,
                ),
            ),
        ),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observational evidence only.",
    )


def _policy(*, max_elapsed_slowdown_ratio: float = 3.1) -> RuntimePerformanceApprovalPolicy:
    return RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=max_elapsed_slowdown_ratio,
        max_peak_process_tree_rss_ratio=2.1,
        minimum_workloads=1,
        minimum_max_timesteps=32,
        reviewed=True,
        review_reference="review-2026-08-09",
    )


def _approval_function() -> Callable[..., RuntimePerformanceEvidence]:
    approve = getattr(runtime_performance, "approve_runtime_performance_evidence", None)
    assert callable(approve), "runtime performance approval API is not implemented"
    return cast(Callable[..., RuntimePerformanceEvidence], approve)


def test_reviewed_policy_materializes_approved_evidence_without_measurement_drift() -> None:
    evidence = _observational_evidence()
    policy = _policy()

    approved = _approval_function()(
        evidence=evidence,
        policy=policy,
        approval_note="Reviewed against the retained performance policy.",
    )

    assert approved.performance_approved is True
    assert approved.approval_policy_digest == policy.digest
    assert approved.approval_note == "Reviewed against the retained performance policy."
    assert approved.runtime_version == evidence.runtime_version
    assert approved.platform == evidence.platform
    assert approved.algorithm == evidence.algorithm
    assert approved.dataset_kind == evidence.dataset_kind
    assert approved.source_digest == evidence.source_digest
    assert approved.workloads == evidence.workloads
    assert approved.digest != evidence.digest


def test_performance_approval_refuses_policy_that_does_not_approve_measurements() -> None:
    evidence = _observational_evidence()
    policy = _policy(max_elapsed_slowdown_ratio=2.9)

    with pytest.raises(
        ValueError,
        match="runtime performance policy does not approve evidence: elapsed_slowdown_ratio_exceeded",
    ):
        _approval_function()(
            evidence=evidence,
            policy=policy,
            approval_note="Must not be emitted.",
        )

    assert evidence.performance_approved is False
    assert evidence.approval_policy_digest is None
