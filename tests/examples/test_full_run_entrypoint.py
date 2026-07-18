from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "binance-multitimeframe"
        / "full_run_entrypoint.py"
    )
    spec = importlib.util.spec_from_file_location("full_run_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Process:
    def __init__(self, return_code: int) -> None:
        self.pid = 4321
        self._polls = iter((None, return_code))
        self.returncode: int | None = None

    def poll(self) -> int | None:
        value = next(self._polls)
        if value is not None:
            self.returncode = value
        return value

    def wait(self) -> int:
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, _signal: int) -> None:
        return None


def test_run_with_heartbeat_records_running_and_terminal_state(tmp_path: Path) -> None:
    module = _load_module()
    heartbeat = tmp_path / "heartbeat.json"
    process = _Process(0)
    timestamps = iter(
        (
            datetime(2026, 7, 18, 0, 0, tzinfo=UTC),
            datetime(2026, 7, 18, 0, 0, 1, tzinfo=UTC),
        )
    )

    result = module.run_with_heartbeat(
        ("python", "worker.py"),
        heartbeat_path=heartbeat,
        generation="generation-1",
        git_commit="a" * 40,
        image_digest="b" * 64,
        phase="develop",
        interval_seconds=0.0,
        popen_factory=lambda _command: process,
        sleeper=lambda _seconds: None,
        now=lambda: next(timestamps),
    )

    assert result == 0
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["state"] == "exited"
    assert payload["exit_code"] == 0
    assert payload["generation"] == "generation-1"
    assert payload["git_commit"] == "a" * 40
    assert payload["image_digest"] == "b" * 64
    assert payload["phase"] == "develop"
    assert payload["pid"] == 4321


def test_run_with_heartbeat_preserves_child_failure(tmp_path: Path) -> None:
    module = _load_module()
    process = _Process(17)
    heartbeat = tmp_path / "heartbeat.json"

    result = module.run_with_heartbeat(
        ("python", "worker.py"),
        heartbeat_path=heartbeat,
        generation="generation-2",
        git_commit="a" * 40,
        image_digest=None,
        phase="train-selected",
        interval_seconds=0.0,
        popen_factory=lambda _command: process,
        sleeper=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert result == 17
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["exit_code"] == 17


def test_build_training_command_is_generation_scoped(tmp_path: Path) -> None:
    module = _load_module()
    command = module.build_training_command(
        python_executable="python",
        repository_root=tmp_path,
        work_root=tmp_path / "var" / "runs" / "generation-3",
        cache_root=tmp_path / "var" / "cache",
        metadata_mode="frozen_snapshot",
        phase="develop",
    )

    assert command[0] == "python"
    assert "run_full_research_hardened.py" in command[1]
    assert command[command.index("--phase") + 1] == "develop"
    assert command[command.index("--work-root") + 1].endswith("generation-3")


def test_entrypoint_rejects_generation_path_escape(tmp_path: Path) -> None:
    module = _load_module()
    try:
        module.resolve_generation_root(tmp_path, "../escape")
    except ValueError as exc:
        assert "generation" in str(exc)
    else:
        raise AssertionError("unsafe generation was accepted")
