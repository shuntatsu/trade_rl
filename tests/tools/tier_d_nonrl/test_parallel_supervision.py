"""Actual Windows pool ownership under a deliberately tiny artificial budget."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from tools.tier_d_nonrl.tier_d_process_tree import (
    ProcessTree,
    terminate_owned_tree,
    windows_process_snapshot,
)
from tools.tier_d_nonrl.tier_d_protocol import market_protocol


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows process snapshots")
def test_actual_supervisor_samples_and_reaps_real_pool_workers(tmp_path, monkeypatch):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    root = tmp_path / "pool-supervision"
    attempt_root = root / "attempt"
    monkeypatch.setattr(module, "ROOT", tmp_path)

    protocol = market_protocol(max_workers=2)
    protocol["resource_proposal"]["rss_bytes"] = 1
    monkeypatch.setattr(module, "market_protocol", lambda **kwargs: protocol)
    assert protocol["concurrency"]["max_workers"] == 2

    script = tmp_path / "manufactured_pool.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import multiprocessing
            import os
            import sys
            import time
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from pathlib import Path

            def occupy(index, root_text):
                root = Path(root_text)
                payload = bytearray(8 * 1024 * 1024)
                for offset in range(0, len(payload), 4096):
                    payload[offset] = 1
                marker = root / f"worker-{index}.json"
                pending = marker.with_suffix(".tmp")
                pending.write_text(
                    json.dumps({"pid": os.getpid(), "index": index})
                )
                pending.replace(marker)
                time.sleep(30)
                return os.getpid()

            def main():
                root = Path(sys.argv[1])
                root.mkdir(parents=True)
                state = {
                    "status": "RUNNING",
                    "arms": [
                        {"state": "RUNNING", "receipt": None},
                        {"state": "SUBMITTED", "receipt": None},
                        {"state": "UNATTEMPTED", "receipt": None},
                        {"state": "UNATTEMPTED", "receipt": None},
                    ],
                }
                (root / "attempt.json").write_text(json.dumps(state))
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                    futures = [pool.submit(occupy, index, str(root)) for index in range(2)]
                    for future in as_completed(futures):
                        future.result()

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )

    original_popen = subprocess.Popen
    unrelated = original_popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
    )
    unrelated_tree = ProcessTree(unrelated.pid)
    unrelated_sample = None
    observed: dict[str, object] = {"samples": []}

    def launch(command, **kwargs):
        process = original_popen(command, **kwargs)
        if isinstance(command, (list, tuple)) and str(script) in map(str, command):
            observed["process"] = process
        return process

    class ReadyTree(ProcessTree):
        def __init__(self, root_pid):
            super().__init__(root_pid)
            observed["tree"] = self

        def sample(self):
            if self.last_sample is None and not observed.get("readiness_checked"):
                deadline = time.monotonic() + 15
                markers = [attempt_root / f"worker-{index}.json" for index in range(2)]
                while not all(path.exists() for path in markers):
                    if time.monotonic() >= deadline:
                        observed["readiness_checked"] = True
                        raise AssertionError(
                            "both pool workers did not publish PID markers"
                        )
                    time.sleep(0.05)
                observed["readiness_checked"] = True
            sample = super().sample()
            observed["tree"] = self
            observed["samples"].append(sample)
            return sample

    def launch_script(mode, target, review_path, **kwargs):
        assert mode == "synthetic"
        assert target == attempt_root
        return [sys.executable, "-B", str(script), str(target)]

    monkeypatch.setattr(module, "worker_command", launch_script)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "ProcessTree", ReadyTree)

    try:
        unrelated_sample = unrelated_tree.sample()
        assert module.supervised_run("synthetic", root, None) == 1

        markers = [
            json.loads((attempt_root / f"worker-{index}.json").read_text())
            for index in range(2)
        ]
        worker_pids = {row["pid"] for row in markers}
        assert len(worker_pids) == 2
        samples = observed["samples"]
        assert samples
        first_sample = samples[0]
        by_pid = {member.pid: member for member in first_sample.members}
        assert worker_pids <= by_pid.keys()
        assert first_sample.root_pid in by_pid
        assert first_sample.root_rss_bytes > 0
        assert first_sample.rss_bytes > sum(
            by_pid[pid].rss_bytes for pid in worker_pids
        )

        report = json.loads((root / "supervisor.json").read_text(encoding="utf-8"))
        assert report["status"] == "FAILED"
        assert "RSS budget exceeded" in report["failure"]
        assert report["limits"]["rss_bytes"] == 1
        assert report["sampled_peak_rss_bytes"] == first_sample.rss_bytes

        state = json.loads((attempt_root / "attempt.json").read_text(encoding="utf-8"))
        assert state["status"] == "FAILED"
        assert [row["state"] for row in state["arms"]] == [
            "INTERRUPTED",
            "INTERRUPTED",
            "UNATTEMPTED",
            "UNATTEMPTED",
        ]
        assert all(row["receipt"] is None for row in state["arms"])

        process = observed["process"]
        assert process.returncode is not None
        sampled_identities = {
            member.identity for sample in samples for member in sample.members
        }
        current_identities = {row.identity for row in windows_process_snapshot()}
        assert sampled_identities.isdisjoint(current_identities)
        assert unrelated.poll() is None
        assert unrelated_sample is not None
        assert not any(
            row.identity in sampled_identities for row in unrelated_sample.members
        )
    finally:
        try:
            process = observed.get("process")
            if process is not None:
                tree = observed.get("tree")
                if tree is None or not tree.owned:
                    tree = ProcessTree(process.pid)
                    tree.sample()
                terminate_owned_tree(process, tree)
        finally:
            terminate_owned_tree(unrelated, unrelated_tree)
