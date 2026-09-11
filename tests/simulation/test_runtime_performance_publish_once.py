from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import pytest

from trade_rl.simulation.diagnostics.runtime_performance import (
    RuntimePerformanceApprovalPolicy,
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)
from trade_rl.simulation.diagnostics.runtime_performance_io import (
    load_runtime_performance_evidence,
    load_runtime_performance_policy,
    write_runtime_performance_evidence,
    write_runtime_performance_policy,
)

_T = TypeVar("_T")


def _policy(max_elapsed_slowdown_ratio: float) -> RuntimePerformanceApprovalPolicy:
    return RuntimePerformanceApprovalPolicy(
        max_elapsed_slowdown_ratio=max_elapsed_slowdown_ratio,
        max_peak_process_tree_rss_ratio=2.0,
        minimum_workloads=1,
        minimum_max_timesteps=8,
        reviewed=True,
        review_reference="concurrency-contract",
    )


def _evidence(source_digest: str) -> RuntimePerformanceEvidence:
    legacy = RuntimePerformanceMeasurement(
        timesteps=8,
        elapsed_seconds=4.0,
        steps_per_second=2.0,
        peak_self_rss_bytes=100,
        peak_children_rss_bytes=0,
        peak_process_tree_rss_bytes=100,
        peak_process_count=1,
    )
    candidate = RuntimePerformanceMeasurement(
        timesteps=8,
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
        source_digest=source_digest,
        workloads=(
            RuntimePerformanceWorkload(
                timesteps=8,
                legacy_authoritative=legacy,
                nautilus_dual_shadow_streaming=candidate,
            ),
        ),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note="Observational evidence only.",
    )


def _assert_concurrent_publish_once(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path: Path,
    left: _T,
    right: _T,
    writer: Callable[[Path, _T], Path],
    loader: Callable[[Path], _T],
) -> None:
    original_replace = Path.replace
    replace_barrier = threading.Barrier(2)

    def synchronized_replace(source: Path, target: str | Path) -> Path:
        if Path(target) == path:
            replace_barrier.wait(timeout=5.0)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)

    def attempt(value: _T) -> BaseException | None:
        try:
            writer(path, value)
        except BaseException as error:
            return error
        return None

    values = (left, right)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(attempt, value) for value in values)
        outcomes = tuple(future.result(timeout=10.0) for future in futures)

    success_indices = tuple(
        index for index, outcome in enumerate(outcomes) if outcome is None
    )
    failures = tuple(outcome for outcome in outcomes if outcome is not None)
    assert len(success_indices) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert loader(path) == values[success_indices[0]]
    assert not tuple(path.parent.glob(f".{path.name}.tmp*"))


def test_runtime_performance_policy_concurrent_drift_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_publish_once(
        monkeypatch,
        path=tmp_path / "policy.json",
        left=_policy(3.0),
        right=_policy(4.0),
        writer=write_runtime_performance_policy,
        loader=load_runtime_performance_policy,
    )


def test_runtime_performance_evidence_concurrent_drift_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_publish_once(
        monkeypatch,
        path=tmp_path / "evidence.json",
        left=_evidence("a" * 64),
        right=_evidence("b" * 64),
        writer=write_runtime_performance_evidence,
        loader=load_runtime_performance_evidence,
    )
