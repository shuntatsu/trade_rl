from __future__ import annotations

import trade_rl.studio.jobs as studio_jobs


def test_windows_pid_start_token_uses_process_creation_time_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        studio_jobs,
        "_windows_process_start_token",
        lambda pid: f"creation:{pid}",
        raising=False,
    )

    assert studio_jobs._pid_start_token(4321, platform_name="nt") == (
        "creation:4321"
    )
