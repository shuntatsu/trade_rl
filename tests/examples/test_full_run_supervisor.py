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


def _completed(
    stdout: str = "", *, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
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
                            "Image": "sha256:" + "b" * 64,
                            "Name": "/trade-rl-full-generation-1",
                            "Config": {
                                "Labels": {
                                    "trade-rl.supervised": "true",
                                    "trade-rl.project": "trade-rl",
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
                                    "trade-rl.project": "trade-rl",
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

    with pytest.raises(RuntimeError, match="exited unsuccessfully"):
        status(
            expected_generation="generation-1",
            expected_git_commit="a" * 40,
            runner=runner,
        )

    assert all(command[:2] != ("docker", "rm") for command in runner.commands)


def _running_inspect(*, image: str = "sha256:" + "b" * 64) -> str:
    return json.dumps(
        [
            {
                "Id": "container-id",
                "Image": image,
                "Name": "/trade-rl-full-generation-1",
                "Config": {
                    "Labels": {
                        "trade-rl.supervised": "true",
                        "trade-rl.project": "trade-rl",
                        "trade-rl.generation": "generation-1",
                        "trade-rl.git-commit": "a" * 40,
                        "trade-rl.source-tree-digest": "c" * 64,
                        "trade-rl.lockfile-digest": "d" * 64,
                    }
                },
                "State": {
                    "Status": "running",
                    "ExitCode": 0,
                    "StartedAt": "2026-07-18T00:00:00Z",
                    "FinishedAt": "0001-01-01T00:00:00Z",
                    "OOMKilled": False,
                },
            }
        ]
    )


def _heartbeat(observed_at: str) -> str:
    return json.dumps(
        {
            "schema_version": "full_run_heartbeat_v1",
            "state": "running",
            "exit_code": None,
            "generation": "generation-1",
            "git_commit": "a" * 40,
            "image_digest": "b" * 64,
            "observed_at": observed_at,
            "phase": "develop",
            "pid": 123,
            "command": ["python", "runner.py"],
        }
    )


def test_status_rejects_stale_heartbeat() -> None:
    status = _namespace()["supervised_run_status"]
    runner = _Runner(
        [
            _completed("trade-rl-full-generation-1\n"),
            _completed(_running_inspect()),
            _completed(_heartbeat("2026-07-18T00:00:00+00:00")),
        ]
    )

    with pytest.raises(RuntimeError, match="heartbeat.*stale"):
        status(
            expected_generation="generation-1",
            expected_git_commit="a" * 40,
            expected_image_digest="b" * 64,
            expected_source_tree_digest="c" * 64,
            expected_lockfile_digest="d" * 64,
            heartbeat_max_age_seconds=60.0,
            now=lambda: __import__("datetime").datetime(
                2026, 7, 18, 0, 2, tzinfo=__import__("datetime").UTC
            ),
            runner=runner,
        )


def test_status_rejects_actual_image_mismatch() -> None:
    status = _namespace()["supervised_run_status"]
    runner = _Runner(
        [
            _completed("trade-rl-full-generation-1\n"),
            _completed(_running_inspect(image="sha256:" + "e" * 64)),
        ]
    )

    with pytest.raises(RuntimeError, match="image digest mismatch"):
        status(
            expected_generation="generation-1",
            expected_git_commit="a" * 40,
            expected_image_digest="b" * 64,
            runner=runner,
        )


def test_stop_reads_logs_before_removing_container() -> None:
    stop = _namespace()["stop_supervised_run"]
    stopped = json.loads(_running_inspect())
    stopped[0]["State"]["Status"] = "exited"
    stopped[0]["State"]["ExitCode"] = 0
    runner = _Runner(
        [
            _completed("trade-rl-full-generation-1\n"),
            _completed(_running_inspect()),
            _completed(),
            _completed(json.dumps(stopped)),
            _completed("retained-log\n"),
            _completed(),
        ]
    )

    evidence = stop(
        remove=True,
        expected_generation="generation-1",
        expected_git_commit="a" * 40,
        runner=runner,
    )

    logs_index = next(
        i
        for i, command in enumerate(runner.commands)
        if command[:2] == ("docker", "logs")
    )
    remove_index = next(
        i
        for i, command in enumerate(runner.commands)
        if command[:2] == ("docker", "rm")
    )
    assert logs_index < remove_index
    assert evidence["container_log_tail"] == "retained-log\n"


def test_expectation_file_makes_absent_container_fail_closed(tmp_path: Path) -> None:
    status = _namespace()["supervised_run_status"]
    expectation = tmp_path / "expected.json"
    expectation.write_text(
        json.dumps(
            {
                "schema_version": "full_run_supervisor_expectation_v1",
                "generation": "generation-1",
                "git_commit": "a" * 40,
                "image_digest": "b" * 64,
                "source_tree_digest": "c" * 64,
                "lockfile_digest": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    runner = _Runner([_completed("")])

    with pytest.raises(RuntimeError, match="expected supervised trainer is absent"):
        status(expectation_path=expectation, runner=runner)
