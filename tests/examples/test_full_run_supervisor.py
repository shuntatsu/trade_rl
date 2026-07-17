from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "binance-multitimeframe" / "full_run_supervisor.py"


def _namespace() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


class _Runner:
    def __init__(self, responses: list[subprocess.CompletedProcess[str]]) -> None:
        self.responses = list(responses)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if not self.responses:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.responses.pop(0)


def _completed(stdout: str = "", *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess((), returncode, stdout=stdout, stderr="")


def test_start_refuses_an_existing_supervised_container() -> None:
    start = _namespace()["start_supervised_run"]
    runner = _Runner([_completed("trade-rl-full-existing\n")])

    with pytest.raises(RuntimeError, match="already exists"):
        start(
            generation="generation-1",
            container_name="trade-rl-full-generation-1",
            git_commit="a" * 40,
            metadata_mode="frozen_snapshot",
            compose_file=Path("compose.training.yaml"),
            runner=runner,
        )


def test_start_labels_and_reports_the_detached_container() -> None:
    start = _namespace()["start_supervised_run"]
    runner = _Runner(
        [
            _completed(""),
            _completed("container-id\n"),
            _completed(
                json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Name": "/trade-rl-full-generation-1",
                            "Config": {
                                "Labels": {
                                    "trade-rl.supervised": "true",
                                    "trade-rl.generation": "generation-1",
                                    "trade-rl.git-commit": "a" * 40,
                                }
                            },
                            "State": {
                                "Status": "running",
                                "ExitCode": 0,
                                "StartedAt": "2026-07-17T09:00:00Z",
                                "FinishedAt": "0001-01-01T00:00:00Z",
                            },
                        }
                    ]
                )
            ),
        ]
    )

    evidence = start(
        generation="generation-1",
        container_name="trade-rl-full-generation-1",
        git_commit="a" * 40,
        metadata_mode="frozen_snapshot",
        compose_file=Path("compose.training.yaml"),
        runner=runner,
    )

    launch = runner.commands[1]
    assert "--label" in launch
    assert "trade-rl.supervised=true" in launch
    assert evidence["state"] == "running"
    assert evidence["container_id"] == "container-id"


def test_status_reports_terminal_exit_without_deleting_evidence() -> None:
    status = _namespace()["supervised_run_status"]
    runner = _Runner(
        [
            _completed("trade-rl-full-generation-1\n"),
            _completed(
                json.dumps(
                    [
                        {
                            "Id": "container-id",
                            "Name": "/trade-rl-full-generation-1",
                            "Config": {
                                "Labels": {
                                    "trade-rl.supervised": "true",
                                    "trade-rl.generation": "generation-1",
                                    "trade-rl.git-commit": "a" * 40,
                                }
                            },
                            "State": {
                                "Status": "exited",
                                "ExitCode": 1,
                                "StartedAt": "2026-07-17T09:00:00Z",
                                "FinishedAt": "2026-07-17T10:00:00Z",
                            },
                        }
                    ]
                )
            ),
        ]
    )

    evidence = status(runner=runner)

    assert evidence["state"] == "exited"
    assert evidence["exit_code"] == 1
    assert all(command[:2] != ("docker", "rm") for command in runner.commands)
