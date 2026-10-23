"""Resource ownership tests; manufactured processes only, no market access."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.tier_d_nonrl.tier_d_process_tree import (
    ProcessTree,
    ProcessTreeSample,
    windows_process_snapshot,
)
from tools.tier_d_nonrl.tier_d_protocol import market_protocol
from tools.tier_d_nonrl.tier_d_records import initialize_attempt, write_json


def api():
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    assert hasattr(module, "ProcessTree"), "descendant RSS ownership is missing"
    return module


def node(pid, parent, created, rss):
    return api().ProcessUsage(pid, parent, created, rss)


def test_snapshot_sums_redirector_and_all_lawful_descendants_only():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    sample = tree.observe(
        (
            node(10, 1, 100, 4),
            node(20, 10, 101, 30),
            node(21, 20, 102, 7),
            node(99, 1, 99, 999),
        )
    )
    assert sample.rss_bytes == 41
    assert sample.root_rss_bytes == 4
    assert [member.pid for member in sample.members] == [10, 20, 21]
    assert [member.pid for member in sample.owned_members] == [10, 20, 21]
    assert sample.ambiguous_members == ()


def test_creation_identity_refuses_pid_reuse_and_attributes_historical_child_ambiguously():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    tree.observe((node(10, 1, 100, 4), node(20, 10, 101, 30)))
    orphan = tree.observe((node(20, 10, 101, 31), node(21, 20, 102, 7)))
    assert (
        orphan.rss_bytes == 38
    )  # Owned live descendants remain owned after stub exit.
    historical_child = node(22, 20, 150, 999)
    reused = tree.observe(
        (node(10, 1, 200, 4), node(20, 1, 201, 1000), historical_child)
    )
    assert reused.members == (historical_child,)
    assert reused.owned_members == ()
    assert reused.ambiguous_members == (historical_child,)


def test_first_snapshot_pid_reuse_never_owns_replacement_or_its_orphan():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    # The process launched as PID 10 was born at tick 100. Its first visible
    # snapshot row is a reused PID, while an older child still names PID 10.
    # The expected identity is supplied by Popen's retained process handle.
    replacement = node(10, 1, 200, 4)
    surviving_child = node(20, 10, 150, 30)

    first = tree.observe((replacement, surviving_child))
    second = tree.observe((surviving_child,))

    assert tree.root_identity is None
    assert tree.owned == {}
    assert first.owned_members == second.owned_members == ()
    assert first.ambiguous_members == (replacement, surviving_child)
    assert second.ambiguous_members == (surviving_child,)


def test_process_tree_uses_only_handle_identity_and_bounds_cim_precision(monkeypatch):
    module = api()
    process_tree_module = importlib.import_module(
        "tools.tier_d_nonrl.tier_d_process_tree"
    )
    launched = SimpleNamespace(pid=10)
    monkeypatch.setattr(
        process_tree_module,
        "_process_identity_from_popen",
        lambda process: (process.pid, 100),
    )
    within_precision = node(10, 1, 110, 4)
    tree = module.ProcessTree.from_popen(launched, provider=lambda: (within_precision,))

    assert tree.expected_root_identity == (10, 100)
    assert tree.sample().owned_members == (within_precision,)
    assert tree.root_identity == within_precision.identity

    def failed_identity(_process):
        raise OSError("handle query failed")

    monkeypatch.setattr(
        process_tree_module, "_process_identity_from_popen", failed_identity
    )
    replacement = node(10, 1, 200, 4)
    unbound = module.ProcessTree.from_popen(launched, provider=lambda: (replacement,))
    sample = unbound.sample()

    assert unbound.expected_root_identity is None
    assert "handle query failed" in unbound.root_identity_error
    assert unbound.root_identity is None
    assert sample.owned_members == ()
    assert unbound.owned == {}


def test_process_tree_rejects_birth_time_outside_cim_precision_tolerance():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    near_miss = node(10, 1, 111, 4)

    sample = tree.observe((near_miss,))

    assert tree.root_identity is None
    assert tree.owned == {}
    assert sample.owned_members == ()


@pytest.mark.parametrize(
    "expected_root_identity",
    (None, (10, 100)),
    ids=("handle-identity-unavailable", "snapshot-is-a-reused-pid"),
)
def test_unverified_visible_root_counts_as_rss_but_not_kill_owned(
    expected_root_identity,
):
    root = node(10, 1, 200, 800_000_000)
    tree = api().ProcessTree(10, expected_root_identity=expected_root_identity)

    sample = tree.observe((root,))

    assert sample.members == (root,)
    assert sample.rss_bytes == 800_000_000
    assert sample.root_rss_bytes == 800_000_000
    assert sample.owned_members == ()
    assert sample.ambiguous_members == (root,)
    assert tree.owned == {}


@pytest.mark.parametrize(
    ("replacement_parent_pid", "owned_pids", "ambiguous_pids"),
    (
        (10, (10, 20), (21,)),
        (30, (10,), (20, 21)),
    ),
    ids=("verified-replacement", "ambiguous-replacement"),
)
def test_process_tree_attributes_pre_reuse_child_of_reused_parent_ambiguously(
    replacement_parent_pid, owned_pids, ambiguous_pids
):
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    root = node(10, 1, 100, 4)
    original_parent = node(20, 10, 101, 30)
    historical_parent = node(30, 10, 102, 10)
    tree.observe((root, original_parent, historical_parent))

    replacement_parent = node(20, replacement_parent_pid, 200, 1000)
    tree.observe((root, replacement_parent))
    historical_child = node(21, 20, 150, 9)
    sample = tree.observe((root, replacement_parent, historical_child))

    assert sample.members == (root, replacement_parent, historical_child)
    assert tuple(member.pid for member in sample.owned_members) == owned_pids
    assert tuple(member.pid for member in sample.ambiguous_members) == ambiguous_pids
    assert sample.rss_bytes == 1013


def test_process_tree_counts_child_first_seen_after_redirector_exit():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    redirector = node(10, 1, 100, 4_400_000)
    tree.observe((redirector,))

    late_child = node(20, 10, 101, 248_800_000)
    too_early_child = node(21, 10, 99, 900_000)
    sample = tree.observe((too_early_child, late_child))

    assert sample.members == (late_child,)
    assert sample.root_rss_bytes == 0
    assert sample.rss_bytes == 248_800_000


def test_process_tree_does_not_attach_child_to_reused_redirector_pid():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    tree.observe((node(10, 1, 100, 4_400_000),))

    replacement = node(10, 1, 200, 1_000_000)
    unrelated_child = node(20, 10, 201, 248_800_000)
    sample = tree.observe((replacement, unrelated_child))

    assert sample.members == ()
    assert sample.rss_bytes == 0


def test_process_tree_attributes_pre_reuse_orphan_as_ambiguous():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    tree.observe((node(10, 1, 100, 4_400_000),))

    replacement = node(10, 1, 200, 1_000_000)
    child_of_historical_parent = node(20, 10, 150, 248_800_000)
    sample = tree.observe((replacement, child_of_historical_parent))

    assert sample.members == (child_of_historical_parent,)
    assert sample.owned_members == ()
    assert sample.ambiguous_members == (child_of_historical_parent,)
    assert sample.rss_bytes == 248_800_000


def test_late_process_attribution_is_rss_only_and_propagates_ambiguity():
    tree = api().ProcessTree(10, expected_root_identity=(10, 100))
    redirector = node(10, 1, 100, 4_400_000)
    tree.observe((redirector,))
    replacement = node(10, 1, 200, 1_000_000)
    assert tree.observe((replacement,)).members == ()

    # The replacement exits before its child is visible. The stale parent PID
    # is enough for RSS attribution, but never proves that child is ours.
    late_child = node(20, 10, 201, 248_800_000)
    late_grandchild = node(21, 20, 202, 3_000_000)
    sample = tree.observe((late_grandchild, late_child))

    assert sample.members == (late_child, late_grandchild)
    assert sample.owned_members == ()
    assert sample.ambiguous_members == (late_child, late_grandchild)
    assert tree.owned == {redirector.identity: 0}


def test_pid_reuse_after_parent_exit_never_grants_kill_authority(monkeypatch):
    module = api()
    redirector = node(10, 1, 100, 4_400_000)
    replacement = node(10, 1, 200, 1_000_000)
    late_child = node(20, 10, 201, 248_800_000)
    tree = module.ProcessTree(
        10, provider=lambda: (redirector,), expected_root_identity=(10, 100)
    )
    tree.sample()
    tree.observe((replacement,))
    tree.provider = lambda: (late_child,)
    kill_payloads = []

    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout):
            assert timeout == 30
            return 0

        def terminate(self):
            pytest.fail("an already-finished root should not be terminated")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: kill_payloads.append(kwargs["input"]),
    )

    with pytest.raises(RuntimeError, match="ambiguous"):
        module.terminate_owned_tree(FinishedProcess(), tree)

    assert kill_payloads == []
    assert tree.last_sample.ambiguous_members == (late_child,)
    assert tree.last_sample.owned_members == ()


def test_visible_reused_parent_with_older_child_blocks_cleanup_verification(
    monkeypatch,
):
    module = api()
    original = node(10, 1, 100, 4_400_000)
    replacement = node(10, 1, 200, 1_000_000)
    child = node(20, 10, 150, 248_800_000)
    tree = module.ProcessTree(
        10,
        provider=lambda: (replacement, child),
        expected_root_identity=(10, 100),
    )
    tree.observe((original,))
    kill_payloads = []

    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout):
            assert timeout == 30
            return 0

        def terminate(self):
            pytest.fail("an already-finished root should not be terminated")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: kill_payloads.append(kwargs["input"]),
    )

    with pytest.raises(RuntimeError, match="ambiguous"):
        module.terminate_owned_tree(FinishedProcess(), tree)

    assert kill_payloads == []
    assert tree.last_sample.ambiguous_members == (child,)
    assert tree.last_sample.owned_members == ()


def test_cleanup_kills_only_verified_members_when_ambiguous_rows_remain(
    monkeypatch,
):
    module = api()
    root = node(10, 1, 100, 4_400_000)
    original_child = node(20, 10, 101, 40_000_000)
    replacement = node(20, 1, 200, 1_000_000)
    historical_child = node(21, 20, 150, 248_800_000)
    tree = module.ProcessTree(10, expected_root_identity=(10, 100))
    tree.observe((root, original_child))
    sampled = tree.observe((root, replacement, historical_child))
    assert sampled.owned_members == (root,)
    assert sampled.ambiguous_members == (historical_child,)

    snapshots = [(root, replacement, historical_child), (historical_child,)]
    tree.provider = lambda: snapshots.pop(0)
    kill_payloads = []

    class FinishedProcess:
        def poll(self):
            return 0

        def wait(self, timeout):
            assert timeout == 30
            return 0

        def terminate(self):
            pytest.fail("an already-finished root should not be terminated")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: kill_payloads.append(kwargs["input"]),
    )

    with pytest.raises(RuntimeError, match="ambiguous"):
        module.terminate_owned_tree(FinishedProcess(), tree)

    assert len(kill_payloads) == 1
    assert [row["pid"] for row in json.loads(kill_payloads[0])] == [root.pid]
    assert tree.last_sample.ambiguous_members == (historical_child,)
    assert tree.last_sample.owned_members == ()


def test_supervisor_does_not_reconcile_while_ambiguous_rss_rows_remain(
    tmp_path, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=os.environ))
    root = tmp_path / "ambiguous-supervisor"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    protocol = market_protocol()
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)

    redirector = node(10, 1, 100, 4_400_000)
    late_child = node(20, 10, 101, 248_800_000)
    snapshots = [(redirector,), (late_child,)]
    cleanup_calls = []

    class FinishedProcess:
        pid = 10
        returncode = 0

        def __enter__(self):
            initialize_attempt(root / "attempt", protocol)
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def terminate(self):
            pytest.fail("the completed worker is already reaped")

        def wait(self, timeout):
            return self.returncode

    process = FinishedProcess()

    class AmbiguousTree(ProcessTree):
        def __init__(self, pid):
            assert pid == process.pid
            super().__init__(pid, provider=lambda: snapshots.pop(0))

    def refuse_ambiguous_cleanup(owned_process, tree):
        assert owned_process is process
        assert tree.last_sample.owned_members == ()
        assert tree.last_sample.ambiguous_members == (late_child,)
        cleanup_calls.append(True)
        raise RuntimeError("ambiguous RSS-only rows remain; refusing cleanup")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module, "ProcessTree", AmbiguousTree)
    monkeypatch.setattr(module, "terminate_owned_tree", refuse_ambiguous_cleanup)

    assert module.supervised_run("synthetic", root, None) == 1
    report = json.loads((root / "supervisor.json").read_text())
    attempt = json.loads((root / "attempt/attempt.json").read_text())
    assert report["status"] == "FAILED"
    assert "ambiguous" in report["failure"]
    assert report["receipt_reconciliation"]["state"] == (
        "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"
    )
    assert report["sampled_peak_rss_bytes"] == late_child.rss_bytes
    progress_records = [
        json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    resource_samples = [
        row for row in progress_records if row["event"] == "resource_sample"
    ]
    assert [row["rss_bytes"] for row in resource_samples] == [
        redirector.rss_bytes,
        late_child.rss_bytes,
    ]
    assert resource_samples[-1]["ambiguous_rss_process_count"] == 1
    assert cleanup_calls == [True]
    assert all(row["state"] == "UNATTEMPTED" for row in attempt["arms"])


@pytest.mark.parametrize(
    ("root_observed", "expected_status"),
    (
        pytest.param(True, "COMPLETED", id="verified-root"),
        pytest.param(False, "FAILED", id="root-identity-never-observed"),
    ),
)
def test_supervisor_requires_observed_root_before_successful_completion(
    tmp_path, monkeypatch, root_observed, expected_status
):
    module = api()
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    root = tmp_path / "root-identity"
    protocol = market_protocol()
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)
    root_row = node(10, 1, 100, 4096)
    samples = iter(((root_row,), ()) if root_observed else ((), ()))

    class FinishedProcess:
        pid = 10
        returncode = 0

        def __enter__(self):
            initialize_attempt(root / "attempt", protocol)
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def terminate(self):
            pytest.fail("the completed worker is already reaped")

        def wait(self, timeout):
            return self.returncode

    process = FinishedProcess()

    class ObservedTree(ProcessTree):
        @classmethod
        def from_popen(cls, actual_process):
            tree = cls(actual_process.pid)
            tree.expected_root_identity = (actual_process.pid, 100)
            return tree

        def __init__(self, pid):
            assert pid == process.pid
            super().__init__(
                pid,
                provider=lambda: next(samples, ()),
            )

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module, "ProcessTree", ObservedTree)

    result = module.supervised_run("synthetic", root, None)

    report = json.loads((root / "supervisor.json").read_text())
    assert report["status"] == expected_status
    assert result == (0 if root_observed else 1)
    if root_observed:
        assert report["failure"] is None
    else:
        assert "root process birth identity was never observed" in report["failure"]
        assert report["receipt_reconciliation"]["state"] == (
            "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"
        )


def test_nonzero_child_exit_finalizes_failed_without_invented_arm_receipts(tmp_path):
    tmp_path = tmp_path / "attempt"
    state = initialize_attempt(tmp_path, market_protocol())
    assert state["status"] == "PREPARING"
    api().mark_attempt_failure(tmp_path, failure=None, returncode=7)
    final = json.loads((tmp_path / "attempt.json").read_text())
    assert final["status"] == "FAILED"
    assert "7" in final["supervisor_failure"]
    assert len(final["arms"]) == 48
    assert all(
        row["state"] == "UNATTEMPTED" and row["receipt"] is None
        for row in final["arms"]
    )


@pytest.mark.skipif(os.name != "nt", reason="actual Windows venv redirector validation")
def test_actual_windows_redirector_memory_and_owned_leaf_termination(tmp_path):
    module = api()
    ready = tmp_path / "leaf-ready.txt"
    leaf_code = (
        "import pathlib,sys,time; "
        "payload=bytearray(24*1024*1024); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); "
        "time.sleep(30)"
    )
    root_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-B','-c',sys.argv[2],sys.argv[1]]); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", root_code, str(ready), leaf_code]
    )
    tree = module.ProcessTree.from_popen(process)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        sample = tree.sample()
        assert len(sample.members) >= 2
        assert tree.expected_root_identity is not None
        root_member = next(
            member for member in sample.owned_members if member.pid == process.pid
        )
        assert tree.root_identity == root_member.identity
        assert tree.expected_root_identity[0] == process.pid
        assert abs(tree.expected_root_identity[1] - root_member.creation_ticks) <= 10
        assert sample.rss_bytes > sample.root_rss_bytes + 24 * 1024 * 1024
        write_json(
            tmp_path / "actual-process-tree.json",
            {
                "scope": "ARTIFICIAL_RESOURCE_CONTROL_ONLY",
                "rss_bytes": sample.rss_bytes,
                "root_rss_bytes": sample.root_rss_bytes,
                "members": [asdict(member) for member in sample.members],
            },
        )
    finally:
        module.terminate_owned_tree(process, tree)
        process.wait(timeout=10)
    assert tree.sample().members == ()
    observation = json.loads((tmp_path / "actual-process-tree.json").read_text())
    observation.update(
        termination_verified=True,
        owned_members_remaining=0,
        child_returncode=process.returncode,
    )
    write_json(tmp_path / "actual-process-tree.json", observation, replace=True)


def test_supervisor_calls_owned_cleanup_after_repeated_snapshot_failure(
    tmp_path, monkeypatch
):
    module = api()
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=os.environ))
    root = tmp_path / "supervised-fake-cim-failure"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    protocol = market_protocol()
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)
    owned_cleanup_calls = []

    class FakeProcess:
        pid = 10
        returncode = None
        leaf_alive = True
        reaped = False

        def __enter__(self):
            initialize_attempt(root / "attempt", protocol)
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("artificial worker", timeout)
            self.reaped = True
            return self.returncode

        def terminate(self):
            self.returncode = 7

    class FailingTree:
        calls = 0

        @classmethod
        def from_popen(cls, process):
            return cls(process.pid)

        def __init__(self, pid):
            self.last_sample = ProcessTreeSample(
                (node(10, 1, 100, 4), node(20, 10, 101, 30)),
                10,
                owned_members=(node(10, 1, 100, 4), node(20, 10, 101, 30)),
                ambiguous_members=(),
            )

        def sample(self):
            self.calls += 1
            if self.calls == 1:
                return self.last_sample
            raise OSError("artificial CIM provider unavailable")

    process = FakeProcess()

    def cleanup(process, tree):
        owned_cleanup_calls.append(True)
        assert len(tree.last_sample.members) == 2
        process.leaf_alive = False
        process.terminate()
        process.wait(timeout=1)

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(module, "ProcessTree", FailingTree)
    monkeypatch.setattr(module, "terminate_owned_tree", cleanup)
    assert module.supervised_run("synthetic", root, None) == 1
    assert owned_cleanup_calls == [True]
    assert not process.leaf_alive and process.reaped
    state = json.loads((root / "attempt/attempt.json").read_text())
    assert state["status"] == "FAILED"
    assert all(row["state"] == "UNATTEMPTED" for row in state["arms"])


@pytest.mark.skipif(os.name != "nt", reason="actual failed-monitor Windows cleanup")
def test_actual_supervisor_cim_failure_stops_owned_leaf_keeps_unrelated(
    tmp_path, monkeypatch
):
    module = api()
    root = tmp_path / "supervised-actual-cim-failure"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    original_popen = subprocess.Popen
    observed = {}
    child_code = (
        "import json,pathlib,sys,time; "
        "payload=bytearray(24*1024*1024); "
        "root=pathlib.Path(sys.argv[1]); root.mkdir(); "
        "root.joinpath('attempt.json').write_text(json.dumps({'status':'PREPARING','arms':"
        "[{'state':'UNATTEMPTED','receipt':None} for _ in range(48)]})); "
        "root.joinpath('ready.txt').write_text('ready'); time.sleep(30)"
    )
    unrelated = original_popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
    )
    unrelated_tree = ProcessTree.from_popen(unrelated)
    unrelated_tree.sample()

    def launch(command, **kwargs):
        if len(command) >= 4 and command[2:4] == ["-m", "tools.tier_d_nonrl"]:
            worker = original_popen(
                [sys.executable, "-B", "-c", child_code, str(root / "attempt")],
                **kwargs,
            )
            observed["worker"] = worker
            return worker
        return original_popen(command, **kwargs)

    class FailingTree(ProcessTree):
        def sample(self):
            if self.last_sample is not None:
                raise OSError("injected persistent CIM monitoring failure")
            deadline = time.monotonic() + 10
            while (
                not (root / "attempt/ready.txt").exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            sample = super().sample()
            observed["owned"] = sample.members
            return sample

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "ProcessTree", FailingTree)
    try:
        assert module.supervised_run("synthetic", root, None) == 1
        remaining = {row.identity for row in windows_process_snapshot()}
        leftovers = [
            row.identity for row in observed["owned"] if row.identity in remaining
        ]
        assert leftovers == []
        assert observed["worker"].returncode is not None
        assert unrelated.poll() is None
        state = json.loads((root / "attempt/attempt.json").read_text())
        assert state["status"] == "FAILED"
        assert all(
            row["state"] == "UNATTEMPTED" and row["receipt"] is None
            for row in state["arms"]
        )
        write_json(
            root / "failure-cleanup-proof.json",
            {
                "scope": "ARTIFICIAL_RESOURCE_FAILURE_ONLY",
                "owned_members_remaining": 0,
                "unrelated_alive_after_owned_cleanup": True,
                "worker_reaped": True,
                "initial_owned_members": [asdict(row) for row in observed["owned"]],
            },
        )
    finally:
        worker = observed.get("worker")
        if worker is not None and worker.poll() is None:
            real_tree = ProcessTree.from_popen(worker)
            real_tree.sample()
            module.terminate_owned_tree(worker, real_tree)
        module.terminate_owned_tree(unrelated, unrelated_tree)
