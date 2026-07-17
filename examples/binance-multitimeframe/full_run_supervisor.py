#!/usr/bin/env python3
"""Start, inspect, and stop one durable full-research Docker container."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_success(
    completed: subprocess.CompletedProcess[str],
    *,
    command: Sequence[str],
) -> str:
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {command!r}: {detail}")
    return completed.stdout.strip()


def _supervised_names(runner: CommandRunner) -> tuple[str, ...]:
    command = (
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=trade-rl.supervised=true",
        "--format",
        "{{.Names}}",
    )
    output = _require_success(runner(command), command=command)
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _inspect(name: str, runner: CommandRunner) -> dict[str, Any]:
    command = ("docker", "inspect", name)
    output = _require_success(runner(command), command=command)
    payload = json.loads(output)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("docker inspect returned an invalid payload")
    item = dict(payload[0])
    state = item.get("State")
    config = item.get("Config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise RuntimeError("docker inspect lacks state or configuration")
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        labels = {}
    raw_name = item.get("Name")
    resolved_name = str(raw_name).removeprefix("/")
    container_id = str(item.get("Id", ""))
    if not container_id or not resolved_name:
        raise RuntimeError("docker inspect lacks container identity")
    return {
        "container_id": container_id,
        "container_name": resolved_name,
        "exit_code": int(state.get("ExitCode", 0)),
        "finished_at": str(state.get("FinishedAt", "")),
        "generation": str(labels.get("trade-rl.generation", "")),
        "git_commit": str(labels.get("trade-rl.git-commit", "")),
        "metadata_mode": str(labels.get("trade-rl.metadata-mode", "")),
        "observed_at": datetime.now(UTC).isoformat(),
        "schema_version": "full_run_supervisor_status_v1",
        "started_at": str(state.get("StartedAt", "")),
        "state": str(state.get("Status", "unknown")),
        "supervised": labels.get("trade-rl.supervised") == "true",
    }


def _validate_identity(
    *,
    generation: str,
    container_name: str,
    git_commit: str,
    metadata_mode: str,
) -> None:
    if not _NAME.fullmatch(generation):
        raise ValueError("generation must be a safe non-empty identifier")
    if not _NAME.fullmatch(container_name):
        raise ValueError("container_name must be a safe non-empty identifier")
    if _GIT_SHA.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a lowercase 40-character SHA")
    if metadata_mode not in {
        "historical_signed",
        "frozen_snapshot",
        "conservative_static",
    }:
        raise ValueError("metadata_mode is unsupported")


def start_supervised_run(
    *,
    generation: str,
    container_name: str,
    git_commit: str,
    metadata_mode: str,
    compose_file: Path,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Start exactly one labeled detached trainer and return inspected evidence."""

    _validate_identity(
        generation=generation,
        container_name=container_name,
        git_commit=git_commit,
        metadata_mode=metadata_mode,
    )
    existing = _supervised_names(runner)
    if existing:
        raise RuntimeError(
            "a supervised full trainer already exists: " + ", ".join(existing)
        )
    command = (
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--detach",
        "--name",
        container_name,
        "--label",
        "trade-rl.supervised=true",
        "--label",
        f"trade-rl.generation={generation}",
        "--label",
        f"trade-rl.git-commit={git_commit}",
        "--label",
        f"trade-rl.metadata-mode={metadata_mode}",
        "trainer",
    )
    _require_success(runner(command), command=command)
    evidence = _inspect(container_name, runner)
    if evidence["state"] != "running" or evidence["supervised"] is not True:
        raise RuntimeError("supervised trainer did not enter the running state")
    return evidence


def supervised_run_status(
    *,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Inspect the one retained supervised container without removing evidence."""

    names = _supervised_names(runner)
    if not names:
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "schema_version": "full_run_supervisor_status_v1",
            "state": "absent",
            "supervised": False,
        }
    if len(names) != 1:
        raise RuntimeError("multiple supervised full trainers exist: " + ", ".join(names))
    return _inspect(names[0], runner)


def stop_supervised_run(
    *,
    remove: bool = False,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Stop the retained trainer and optionally remove it after capturing status."""

    names = _supervised_names(runner)
    if not names:
        return supervised_run_status(runner=runner)
    if len(names) != 1:
        raise RuntimeError("multiple supervised full trainers exist: " + ", ".join(names))
    name = names[0]
    inspect_before = _inspect(name, runner)
    if inspect_before["state"] == "running":
        command = ("docker", "stop", "--time", "60", name)
        _require_success(runner(command), command=command)
    evidence = _inspect(name, runner)
    if remove:
        command = ("docker", "rm", name)
        _require_success(runner(command), command=command)
        evidence = {**evidence, "removed": True}
    return evidence


def _write_evidence(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("start", "status", "stop"))
    parser.add_argument("--generation")
    parser.add_argument("--container-name")
    parser.add_argument("--git-commit")
    parser.add_argument("--metadata-mode", default="frozen_snapshot")
    parser.add_argument("--compose-file", type=Path, default=Path("compose.training.yaml"))
    parser.add_argument("--evidence-path", type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.operation == "start":
        if not args.generation or not args.container_name or not args.git_commit:
            parser.error("start requires --generation, --container-name, and --git-commit")
        payload = start_supervised_run(
            generation=args.generation,
            container_name=args.container_name,
            git_commit=args.git_commit,
            metadata_mode=args.metadata_mode,
            compose_file=args.compose_file,
        )
    elif args.operation == "status":
        payload = supervised_run_status()
    else:
        payload = stop_supervised_run(remove=args.remove)
    if args.evidence_path is not None:
        _write_evidence(args.evidence_path, payload)
    print(json.dumps(payload, sort_keys=True))
    state = payload.get("state") if isinstance(payload, dict) else None
    if state == "exited" and int(payload.get("exit_code", 1)) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
