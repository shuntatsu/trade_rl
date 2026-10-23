"""Manufactured supervisor clocks; no Dataset or financial execution."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tools.tier_d_nonrl.tier_d_process_tree import ProcessTree, windows_process_snapshot
from tools.tier_d_nonrl.tier_d_protocol import market_protocol
from tools.tier_d_nonrl.tier_d_records import initialize_attempt


@pytest.mark.parametrize("mode", ["stalled", "advancing", "preparing", "malformed"])
def test_supervisor_polls_native_progress_independently_of_resource_samples(
    tmp_path, monkeypatch, mode
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    root = tmp_path / mode
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", environ=os.environ))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    protocol = market_protocol()
    protocol["progress_policy"]["no_progress_seconds"] = 3
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)
    samples = []
    monkeypatch.setattr(
        module, "process_rss", lambda process: samples.append(True) or 1
    )
    polls = []

    class Reader:
        states = {}

        def __init__(self, *args, **kwargs):
            assert kwargs["expected_policy"] == protocol["progress_policy"]
            assert kwargs.get("expected_protocol") is None

        def poll(self):
            polls.append(clock.now)
            if mode == "malformed":
                raise ValueError("manufactured foreign journal")
            self.states = (
                {"arm": {"count": int(clock.now)}} if mode != "preparing" else {}
            )
            return iter(())

        def stalled(self):
            return (
                [{"arm_id": "arm", "count": 0, "total": 10, "idle_seconds": clock.now}]
                if mode == "stalled" and clock.now >= 3
                else []
            )

    monkeypatch.setattr(module, "ProgressReader", Reader, raising=False)

    class Process:
        pid = 10
        returncode = None

        def __enter__(self):
            initialize_attempt(root / "attempt", protocol)
            return self

        def __exit__(self, *args):
            return False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is not None:
                return self.returncode
            clock.now += min(timeout, 8 - clock.now)
            if clock.now >= 8:
                self.returncode = 0
                return 0
            raise subprocess.TimeoutExpired("manufactured worker", timeout)

        def terminate(self):
            self.returncode = 7

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    expected_failure = mode in {"stalled", "malformed"}
    assert module.supervised_run("synthetic", root, None) == int(expected_failure)
    report = json.loads((root / "supervisor.json").read_text())
    assert len(polls) >= 1
    assert samples == [], (
        "one-second progress polling must not force ten-second RSS scans"
    )
    if expected_failure:
        assert clock.now <= 3
        assert process.returncode == 7
        assert report["failure"]
        if mode == "stalled":
            assert "native progress stalled" in report["failure"]
            assert report["stalled_arms"][0]["arm_id"] == "arm"
    else:
        assert clock.now == 8
        assert report["status"] == "COMPLETED"
    assert report["progress_policy"] == protocol["progress_policy"]


@pytest.mark.skipif(os.name != "nt", reason="actual Windows no-progress cleanup")
def test_actual_stalled_journal_stops_owned_child_and_retains_verified_prefix(
    tmp_path, monkeypatch
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    root = tmp_path / "actual-progress-stop"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    protocol = market_protocol()
    protocol.update(
        scope="ARTIFICIAL_PROGRESS_PROCESS_ONLY",
        accounts=["A"],
        scenarios=["BASEL0"],
        configurations=[{"id": "fixture"}],
        actor={"start": 1, "stop": 2},
        terminal={"stop_inclusive": 3},
    )
    protocol["progress_policy"]["no_progress_seconds"] = 1
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)
    child_code = """
import json,pathlib,sys,time
from tools.tier_d_nonrl.progress import ArmProgressSpec
from tools.tier_d_nonrl.progress_io import ProgressJournal,publish_roster
from tools.tier_d_nonrl.tier_d_records import initialize_attempt,write_json
root=pathlib.Path(sys.argv[1]); protocol=json.loads(sys.argv[2])
state=initialize_attempt(root,protocol)
specs=tuple(ArmProgressSpec('fixture','A','BASEL0',role,digest*64,1,2,3)
            for role,digest in [('nonrl','a'),('cash','b')])
publish_roster(root,protocol,specs)
state['arms'][0]['state']='SUBMITTED'
write_json(root/'attempt.json',state,replace=True)
with ProgressJournal(root,specs[0]) as journal:
    journal.advance('actor',1,2)
    (root/'ready').write_text('ready')
    time.sleep(30)
"""
    original_popen = subprocess.Popen
    unrelated = original_popen(
        [sys.executable, "-B", "-c", "import time;time.sleep(30)"]
    )
    unrelated_tree = ProcessTree.from_popen(unrelated)
    unrelated_tree.sample()
    observed = {}

    def launch(command, **kwargs):
        if command[2:4] == ["-m", "tools.tier_d_nonrl"]:
            worker = original_popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    child_code,
                    str(root / "attempt"),
                    json.dumps(protocol),
                ],
                **kwargs,
            )
            observed["worker"] = worker
            return worker
        return original_popen(command, **kwargs)

    class ReadyTree(ProcessTree):
        def sample(self):
            if self.last_sample is None:
                deadline = time.monotonic() + 10
                while (
                    not (root / "attempt/ready").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                assert (root / "attempt/ready").exists()
            sample = super().sample()
            observed.setdefault("owned", sample.members)
            return sample

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "ProcessTree", ReadyTree)
    try:
        assert module.supervised_run("synthetic", root, None) == 1
        report = json.loads((root / "supervisor.json").read_text())
        assert "native progress stalled" in report["failure"]
        assert [(row["count"], row["target"]) for row in report["stalled_arms"]] == [
            (1, 2)
        ]
        lines = list((root / "attempt/native-progress").glob("*.jsonl"))
        assert len(lines) == 1
        prefix = [json.loads(line) for line in lines[0].read_text().splitlines()]
        assert [row["event"] for row in prefix] == ["native_started", "native_advanced"]
        assert prefix[-1]["count"] == 1
        state = json.loads((root / "attempt/attempt.json").read_text())
        assert [row["state"] for row in state["arms"]] == ["INTERRUPTED", "UNATTEMPTED"]
        assert all(row["receipt"] is None for row in state["arms"])
        remaining = {row.identity for row in windows_process_snapshot()}
        assert not any(row.identity in remaining for row in observed["owned"])
        assert observed["worker"].returncode is not None
        assert unrelated.poll() is None
    finally:
        worker = observed.get("worker")
        if worker is not None and worker.poll() is None:
            tree = ProcessTree.from_popen(worker)
            tree.sample()
            module.terminate_owned_tree(worker, tree)
        module.terminate_owned_tree(unrelated, unrelated_tree)
