"""Full supervisor finalization tests with a manufactured owned process tree."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.tools.tier_d_nonrl.test_stopped_receipts import (
    stopped_attempt as _source_stopped_attempt,
)
from tools.tier_d_nonrl.tier_d_process_tree import ProcessTreeSample, ProcessUsage


@pytest.fixture(scope="module")
def stopped_attempt_fixture(tmp_path_factory):
    return _source_stopped_attempt.__wrapped__(tmp_path_factory)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _install_fake_owned_supervisor(
    monkeypatch,
    *,
    root: Path,
    fixture_source: Path,
    max_workers: int = 1,
    tree_identity: bool = True,
    cleanup_failure: bool = False,
    source_change: str | None = None,
):
    supervisor = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    monkeypatch.setattr(
        supervisor, "os", SimpleNamespace(name="nt", environ=os.environ)
    )
    monkeypatch.setattr(
        supervisor,
        "subprocess",
        SimpleNamespace(
            STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
        ),
    )
    monkeypatch.setattr(supervisor, "ROOT", root.parent)
    actual_protocol = supervisor.market_protocol
    protocol = actual_protocol(max_workers=max_workers)
    monkeypatch.setattr(supervisor, "market_protocol", lambda **kwargs: protocol)

    actual_head = supervisor.source_head
    actual_pins = supervisor.source_pins
    stopped = False

    def source_head():
        if source_change == "head" and stopped:
            return "0" * 40
        return actual_head()

    def source_pins():
        if source_change == "pins" and stopped:
            return {"changed-driver.py": "0" * 64}
        return actual_pins()

    monkeypatch.setattr(supervisor, "source_head", source_head)
    monkeypatch.setattr(supervisor, "source_pins", source_pins)

    root_usage = ProcessUsage(pid=4100, parent_pid=1, creation_ticks=100, rss_bytes=8)
    child_usage = ProcessUsage(
        pid=4101, parent_pid=4100, creation_ticks=101, rss_bytes=12
    )
    sample = ProcessTreeSample(
        (root_usage, child_usage),
        root_pid=4100,
        owned_members=(root_usage, child_usage),
        ambiguous_members=(),
    )

    class FakeProcess:
        pid = 4100
        returncode = 7

        def __init__(self):
            self.reaped = False
            self.wait_calls: list[float] = []

        def __enter__(self):
            shutil.copytree(fixture_source, root / "attempt")
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            self.reaped = True
            return self.returncode

        def terminate(self):
            self.returncode = 7

    process = FakeProcess()
    trees = []

    class FakeTree:
        def __init__(self, pid):
            assert pid == process.pid
            self.root_identity = root_usage.identity if tree_identity else None
            self.owned = (
                {root_usage.identity: 0, child_usage.identity: 1}
                if tree_identity
                else {}
            )
            self.last_sample = (
                sample
                if tree_identity
                else ProcessTreeSample((), pid, owned_members=(), ambiguous_members=())
            )
            self.stopped = False
            trees.append(self)

        def sample(self):
            return self.last_sample

    cleanup_calls = []

    def cleanup(owned_process, tree):
        nonlocal stopped
        cleanup_calls.append((owned_process, tree))
        assert owned_process is process
        if cleanup_failure:
            raise OSError("manufactured owned-tree cleanup failure")
        tree.stopped = True
        process.wait(timeout=30)
        stopped = True

    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
        raising=False,
    )
    monkeypatch.setattr(supervisor, "ProcessTree", FakeTree)
    monkeypatch.setattr(supervisor, "terminate_owned_tree", cleanup)

    return supervisor, records, protocol, process, trees, cleanup_calls


def test_supervised_run_reconciles_after_verified_owned_tree_stop(
    tmp_path, monkeypatch, stopped_attempt_fixture
):
    fixture_source, _ = stopped_attempt_fixture
    fixture_before = _snapshot(fixture_source)
    root = tmp_path / "supervised-success"
    (
        supervisor,
        records,
        _protocol,
        process,
        trees,
        cleanup_calls,
    ) = _install_fake_owned_supervisor(
        monkeypatch, root=root, fixture_source=fixture_source
    )
    publish_calls = []
    publish = supervisor._publish_stopped_receipts

    def observe_publish(*args, **kwargs):
        publish_calls.append(kwargs.copy())
        return publish(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_publish_stopped_receipts", observe_publish)

    assert supervisor.supervised_run("synthetic", root, None) == 1

    report = records.read_json(root / "supervisor.json")
    state = records.read_json(root / "attempt/attempt.json")
    assert report["status"] == "FAILED"
    assert report["receipt_reconciliation"] == {
        "state": "RECONCILED",
        "completed_arms": 1,
    }
    assert state["status"] == "FAILED"
    assert [(row["role"], row["state"]) for row in state["arms"]] == [
        ("nonrl", "COMPLETED"),
        ("cash", "INTERRUPTED"),
    ]
    assert publish_calls == [{"owned_tree_stopped": True}]
    assert cleanup_calls == [(process, trees[0])]
    assert trees[0].stopped and process.reaped and process.wait_calls == [30]

    nonrl_relative = "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    cash_relative = "receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    assert (root / "attempt" / nonrl_relative).read_bytes() == fixture_before[
        nonrl_relative
    ]
    assert not (root / "attempt" / cash_relative).exists()
    assert _snapshot(fixture_source) == fixture_before
    assert records.read_json(
        root / "attempt/output-manifest.json"
    ) == records.output_manifest(root / "attempt")
    assert records.read_json(root / "output-manifest.json") == records.output_manifest(
        root
    )


@pytest.mark.parametrize("stop_proof", ["cleanup_failure", "missing_root_identity"])
def test_supervised_run_skips_reconciliation_without_verified_tree_stop(
    tmp_path, monkeypatch, stopped_attempt_fixture, stop_proof
):
    fixture_source, _ = stopped_attempt_fixture
    fixture_before = _snapshot(fixture_source)
    root = tmp_path / f"supervised-unverified-{stop_proof}"
    supervisor, records, _, _, _, cleanup_calls = _install_fake_owned_supervisor(
        monkeypatch,
        root=root,
        fixture_source=fixture_source,
        tree_identity=stop_proof != "missing_root_identity",
        cleanup_failure=stop_proof == "cleanup_failure",
    )
    publish_calls = []

    def observe_publish(*args, **kwargs):
        publish_calls.append(kwargs.copy())
        return {"state": "unexpected"}

    monkeypatch.setattr(supervisor, "_publish_stopped_receipts", observe_publish)

    assert supervisor.supervised_run("synthetic", root, None) == 1

    report = records.read_json(root / "supervisor.json")
    state = records.read_json(root / "attempt/attempt.json")
    assert report["status"] == "FAILED"
    assert report["receipt_reconciliation"] == {
        "state": "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"
    }
    assert state["status"] == "FAILED"
    assert [(row["role"], row["state"]) for row in state["arms"]] == [
        ("nonrl", "INTERRUPTED"),
        ("cash", "INTERRUPTED"),
    ]
    assert publish_calls == []
    assert len(cleanup_calls) == 1
    assert (
        root / "attempt/receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    ).read_bytes() == fixture_before[
        "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    ]
    assert not (
        root / "attempt/receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    ).exists()
    assert _snapshot(fixture_source) == fixture_before


@pytest.mark.parametrize("change", ["head", "pins", "protocol"])
def test_supervised_run_refuses_changed_source_or_synthetic_protocol(
    tmp_path, monkeypatch, stopped_attempt_fixture, change
):
    fixture_source, _ = stopped_attempt_fixture
    fixture_before = _snapshot(fixture_source)
    root = tmp_path / f"supervised-changed-{change}"
    supervisor, records, _, _, _, _ = _install_fake_owned_supervisor(
        monkeypatch,
        root=root,
        fixture_source=fixture_source,
        max_workers=2 if change == "protocol" else 1,
        source_change=change if change in {"head", "pins"} else None,
    )
    publish_calls = []

    def observe_publish(*args, **kwargs):
        publish_calls.append(kwargs.copy())
        return {"state": "unexpected"}

    monkeypatch.setattr(supervisor, "_publish_stopped_receipts", observe_publish)

    assert supervisor.supervised_run("synthetic", root, None) == 1

    report = records.read_json(root / "supervisor.json")
    state = records.read_json(root / "attempt/attempt.json")
    assert report["status"] == "FAILED"
    assert report["receipt_reconciliation"]["state"] == "REFUSED"
    assert publish_calls == []
    assert state["status"] == "FAILED"
    assert [(row["role"], row["state"]) for row in state["arms"]] == [
        ("nonrl", "INTERRUPTED"),
        ("cash", "INTERRUPTED"),
    ]
    assert (
        root / "attempt/receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    ).read_bytes() == fixture_before[
        "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    ]
    assert not (
        root / "attempt/receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    ).exists()
    assert _snapshot(fixture_source) == fixture_before


def test_supervisor_persists_initial_failure_before_attempt_state_update(
    tmp_path, monkeypatch, stopped_attempt_fixture
):
    fixture_source, _ = stopped_attempt_fixture
    root = tmp_path / "supervised-state-publication-failure"
    supervisor, records, _, _, _, _ = _install_fake_owned_supervisor(
        monkeypatch, root=root, fixture_source=fixture_source
    )
    publish_calls = []

    def observe_publish(*args, **kwargs):
        publish_calls.append(kwargs.copy())
        return {"state": "unexpected"}

    def fail_attempt_state(*args, **kwargs):
        raise OSError("manufactured attempt-state publication failure")

    monkeypatch.setattr(supervisor, "_publish_stopped_receipts", observe_publish)
    monkeypatch.setattr(supervisor, "mark_attempt_failure", fail_attempt_state)

    with pytest.raises(OSError, match="manufactured attempt-state publication"):
        supervisor.supervised_run("synthetic", root, None)

    report_path = root / "supervisor.json"
    assert report_path.is_file(), (
        "initial failure report must survive state-write failure"
    )
    report = records.read_json(report_path)
    assert report["status"] == "FAILED"
    assert report["failure"] == (
        "verified owned descendants remained after worker exit; retain partial attempt"
    )
    assert report["child_returncode"] == 7
    assert "receipt_reconciliation" not in report
    assert publish_calls == []
