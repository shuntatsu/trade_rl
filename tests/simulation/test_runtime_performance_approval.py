from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import trade_rl.simulation.runtime_performance as runtime_performance
import trade_rl.simulation.runtime_performance_io as runtime_performance_io
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)
from trade_rl.simulation.runtime_performance_io import (
    load_runtime_performance_evidence,
    write_runtime_performance_evidence,
    write_runtime_performance_policy,
)


def _measurement(
    *, timesteps: int, elapsed_seconds: float, rss: int
) -> RuntimePerformanceMeasurement:
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


def _policy(
    *, max_elapsed_slowdown_ratio: float = 3.1
) -> RuntimePerformanceApprovalPolicy:
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


def _materialize_function() -> Callable[..., RuntimePerformanceEvidence]:
    materialize = getattr(
        runtime_performance_io,
        "materialize_runtime_performance_approval",
        None,
    )
    assert callable(materialize), (
        "runtime performance approval persistence is not implemented"
    )
    return cast(Callable[..., RuntimePerformanceEvidence], materialize)


def test_reviewed_policy_materializes_approved_evidence_without_measurement_drift() -> (
    None
):
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


def test_performance_approval_refuses_policy_that_does_not_approve_measurements() -> (
    None
):
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


def test_performance_approval_refuses_rebinding_already_approved_evidence() -> None:
    policy = _policy()
    already_approved = replace(
        _observational_evidence(),
        performance_approved=True,
        approval_policy_digest=policy.digest,
        approval_note="Already reviewed.",
    )

    with pytest.raises(
        ValueError,
        match="runtime performance approval requires observational evidence",
    ):
        _approval_function()(
            evidence=already_approved,
            policy=replace(policy, max_elapsed_slowdown_ratio=3.2),
            approval_note="Must not rebind approval provenance.",
        )


def test_materialized_runtime_performance_approval_is_persisted_and_revalidated(
    tmp_path: Path,
) -> None:
    evidence = _observational_evidence()
    policy = _policy()
    evidence_path = tmp_path / "observational.json"
    policy_path = tmp_path / "policy.json"
    approved_path = tmp_path / "approved.json"
    write_runtime_performance_evidence(evidence_path, evidence)
    write_runtime_performance_policy(policy_path, policy)

    approved = _materialize_function()(
        evidence_path=evidence_path,
        policy_path=policy_path,
        output_path=approved_path,
        approval_note="Reviewed representative performance approval.",
    )

    reloaded = load_runtime_performance_evidence(approved_path)
    assert reloaded == approved
    assert approved.performance_approved is True
    assert approved.approval_policy_digest == policy.digest
    assert approved.source_digest == evidence.source_digest
    assert approved.workloads == evidence.workloads
