from __future__ import annotations

import json

import pytest

from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)
from trade_rl.simulation.runtime_performance_io import (
    load_runtime_performance_evidence,
    write_runtime_performance_evidence,
)


def _evidence() -> RuntimePerformanceEvidence:
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
        elapsed_seconds=8.0,
        steps_per_second=1.0,
        peak_self_rss_bytes=90,
        peak_children_rss_bytes=90,
        peak_process_tree_rss_bytes=180,
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
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observational evidence only.",
    )


def test_persisted_runtime_performance_evidence_binds_self_digest(tmp_path) -> None:
    evidence = _evidence()
    path = tmp_path / "runtime-performance-evidence.json"

    write_runtime_performance_evidence(path, evidence)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["evidence_digest"] == evidence.digest
    assert load_runtime_performance_evidence(path) == evidence


def test_runtime_performance_loader_rejects_wrong_self_digest(tmp_path) -> None:
    evidence = _evidence()
    path = tmp_path / "runtime-performance-evidence.json"
    write_runtime_performance_evidence(path, evidence)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence_digest"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence digest mismatch"):
        load_runtime_performance_evidence(path)
