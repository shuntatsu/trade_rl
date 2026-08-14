from __future__ import annotations

import pytest

import trade_rl.studio.jobs as studio_jobs

from .support import FakeProcess


@pytest.fixture(autouse=True)
def isolate_fake_process_tree_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Studio fake workers from crossing into host process-management APIs."""

    original = studio_jobs._terminate_process_tree

    def terminate(process: studio_jobs.ProcessHandle) -> int:
        if isinstance(process, FakeProcess):
            process.terminate()
            return process.wait()
        return original(process)

    monkeypatch.setattr(studio_jobs, "_terminate_process_tree", terminate)
