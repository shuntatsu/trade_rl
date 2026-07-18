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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]
Now = Callable[[], datetime]


def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_success(
    completed: subprocess.CompletedProcess[str], *, command: Sequence[str]
) -> str:
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}: {detail}"
        )
    return completed.stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _validate_sha256(value: str | None, *, field: str) -> None:
    if value is not None and _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")


def _supervised_names(runner: CommandRunner) -> tuple[str, ...]:
    command = (
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=trade-rl.supervised=true",
        "--filter",
        "label=trade-rl.project=trade-rl",
        "--format",
        "{{.Names}}",
    )
    output = _require_success(runner(command), command=command)
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _inspect(name: str, runner: CommandRunner) -> dict[str, Any]:
    command = ("docker", "inspect", name)
    output = _require_success(runner(command), command=command)
    payload = json.loads(output)
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
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
    health_raw = state.get("Health")
    health = health_raw if isinstance(health_raw, dict) else {}
    mounts_raw = item.get("Mounts")
    mounts = mounts_raw if isinstance(mounts_raw, list) else []
    raw_args = item.get("Args")
    args = raw_args if isinstance(raw_args, list) else []
    return {
        "command": {
            "path": str(item.get("Path", "")),
            "args": tuple(str(value) for value in args if isinstance(value, str)),
        },
        "container_id": container_id,
        "container_name": resolved_name,
        "exit_code": int(state.get("ExitCode", 0)),
        "finished_at": str(state.get("FinishedAt", "")),
        "generation": str(labels.get("trade-rl.generation", "")),
        "git_commit": str(labels.get("trade-rl.git-commit", "")),
        "health": str(health.get("Status", "none")),
        "image_digest": str(item.get("Image", "")).removeprefix("sha256:"),
        "lockfile_digest": str(labels.get("trade-rl.lockfile-digest", "")),
        "metadata_mode": str(labels.get("trade-rl.metadata-mode", "")),
        "mounts": tuple(
            {
                "destination": str(mount.get("Destination", "")),
                "name": str(mount.get("Name", "")),
                "source": str(mount.get("Source", "")),
                "type": str(mount.get("Type", "")),
            }
            for mount in mounts
            if isinstance(mount, dict)
        ),
        "observed_at": _utc_now().isoformat(),
        "oom_killed": state.get("OOMKilled") is True,
        "project": str(labels.get("trade-rl.project", "")),
        "schema_version": "full_run_supervisor_status_v2",
        "source_tree_digest": str(labels.get("trade-rl.source-tree-digest", "")),
        "started_at": str(state.get("StartedAt", "")),
        "state": str(state.get("Status", "unknown")),
        "state_error": str(state.get("Error", "")),
        "supervised": labels.get("trade-rl.supervised") == "true",
    }


def _read_heartbeat(
    name: str, *, generation: str, runner: CommandRunner
) -> dict[str, Any]:
    command = (
        "docker",
        "exec",
        name,
        "cat",
        f"/workspace/var/runs/{generation}/heartbeat.json",
    )
    raw = _require_success(runner(command), command=command)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("supervised trainer heartbeat is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("supervised trainer heartbeat must be an object")
    if payload.get("schema_version") != "full_run_heartbeat_v1":
        raise RuntimeError("supervised trainer heartbeat schema is unsupported")
    if payload.get("generation") != generation:
        raise RuntimeError("supervised trainer heartbeat generation mismatch")
    if payload.get("state") != "running":
        raise RuntimeError("running trainer heartbeat is not in running state")
    return dict(payload)


def _load_expectation(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"supervisor expectation is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("supervisor expectation is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("supervisor expectation must be an object")
    if payload.get("schema_version") != "full_run_supervisor_expectation_v1":
        raise RuntimeError("supervisor expectation schema is unsupported")
    generation = payload.get("generation")
    git_commit = payload.get("git_commit")
    image_digest = payload.get("image_digest")
    source_digest = payload.get("source_tree_digest")
    lock_digest = payload.get("lockfile_digest")
    if not isinstance(generation, str) or _NAME.fullmatch(generation) is None:
        raise RuntimeError("supervisor expectation generation is invalid")
    if not isinstance(git_commit, str) or _GIT_SHA.fullmatch(git_commit) is None:
        raise RuntimeError("supervisor expectation git commit is invalid")
    for field, value in (
        ("image_digest", image_digest),
        ("source_tree_digest", source_digest),
        ("lockfile_digest", lock_digest),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeError(f"supervisor expectation {field} is invalid")
    assert isinstance(image_digest, str)
    assert isinstance(source_digest, str)
    assert isinstance(lock_digest, str)
    return {
        "generation": generation,
        "git_commit": git_commit,
        "image_digest": image_digest,
        "source_tree_digest": source_digest,
        "lockfile_digest": lock_digest,
    }


def _resolve_expectations(
    *,
    expectation_path: Path | None,
    expected_generation: str | None,
    expected_git_commit: str | None,
    expected_image_digest: str | None,
    expected_source_tree_digest: str | None,
    expected_lockfile_digest: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    if expectation_path is None:
        return (
            expected_generation,
            expected_git_commit,
            expected_image_digest,
            expected_source_tree_digest,
            expected_lockfile_digest,
        )
    stored = _load_expectation(expectation_path)
    pairs = (
        ("generation", expected_generation),
        ("git_commit", expected_git_commit),
        ("image_digest", expected_image_digest),
        ("source_tree_digest", expected_source_tree_digest),
        ("lockfile_digest", expected_lockfile_digest),
    )
    for field, explicit in pairs:
        if explicit is not None and explicit != stored[field]:
            raise RuntimeError(
                f"explicit {field} conflicts with supervisor expectation"
            )
    return (
        stored["generation"],
        stored["git_commit"],
        stored["image_digest"],
        stored["source_tree_digest"],
        stored["lockfile_digest"],
    )


def _require_expected_identity(
    evidence: dict[str, Any],
    *,
    expected_generation: str | None,
    expected_git_commit: str | None,
    expected_image_digest: str | None = None,
    expected_source_tree_digest: str | None = None,
    expected_lockfile_digest: str | None = None,
) -> None:
    if evidence.get("supervised") is not True or evidence.get("project") != "trade-rl":
        raise RuntimeError("container is not a trade-rl supervised trainer")
    if (
        expected_generation is not None
        and evidence.get("generation") != expected_generation
    ):
        raise RuntimeError("supervised trainer generation mismatch")
    if (
        expected_git_commit is not None
        and evidence.get("git_commit") != expected_git_commit
    ):
        raise RuntimeError("supervised trainer git commit mismatch")
    if (
        expected_image_digest is not None
        and evidence.get("image_digest") != expected_image_digest
    ):
        raise RuntimeError("supervised trainer image digest mismatch")
    if (
        expected_source_tree_digest is not None
        and evidence.get("source_tree_digest") != expected_source_tree_digest
    ):
        raise RuntimeError("supervised trainer source tree digest mismatch")
    if (
        expected_lockfile_digest is not None
        and evidence.get("lockfile_digest") != expected_lockfile_digest
    ):
        raise RuntimeError("supervised trainer lockfile digest mismatch")


def _require_valid_runtime_state(evidence: dict[str, Any]) -> None:
    if evidence.get("oom_killed") is True:
        raise RuntimeError("supervised trainer was OOM killed")
    if evidence.get("health") == "unhealthy":
        raise RuntimeError("supervised trainer health check failed")
    state = evidence.get("state")
    if state in {"dead", "paused", "restarting", "removing", "created", "unknown"}:
        raise RuntimeError(f"supervised trainer entered invalid state: {state}")
    if state == "exited" and int(evidence.get("exit_code", 1)) != 0:
        raise RuntimeError("supervised trainer exited unsuccessfully")


def _container_logs(name: str, runner: CommandRunner) -> str:
    command = ("docker", "logs", "--tail", "200", name)
    completed = runner(command)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return f"<docker logs unavailable: {detail}>"
    return completed.stdout


def _validate_identity(
    *, generation: str, container_name: str, git_commit: str, metadata_mode: str
) -> None:
    if _NAME.fullmatch(generation) is None:
        raise ValueError("generation must be a safe non-empty identifier")
    if _NAME.fullmatch(container_name) is None:
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
    source_tree_digest: str | None = None,
    lockfile_digest: str | None = None,
    expected_image_digest: str | None = None,
    expectation_path: Path | None = None,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Start exactly one labeled detached trainer and retain external expectations."""

    _validate_identity(
        generation=generation,
        container_name=container_name,
        git_commit=git_commit,
        metadata_mode=metadata_mode,
    )
    for field, value in (
        ("source_tree_digest", source_tree_digest),
        ("lockfile_digest", lockfile_digest),
        ("expected_image_digest", expected_image_digest),
    ):
        _validate_sha256(value, field=field)
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
        "trade-rl.project=trade-rl",
        "--label",
        f"trade-rl.generation={generation}",
        "--label",
        f"trade-rl.git-commit={git_commit}",
        "--label",
        f"trade-rl.metadata-mode={metadata_mode}",
        "--label",
        f"trade-rl.source-tree-digest={source_tree_digest or ''}",
        "--label",
        f"trade-rl.lockfile-digest={lockfile_digest or ''}",
        "trainer",
    )
    _require_success(runner(command), command=command)
    evidence = _inspect(container_name, runner)
    actual_image = str(evidence.get("image_digest", ""))
    resolved_image = expected_image_digest or actual_image
    if _SHA256.fullmatch(resolved_image) is None:
        raise RuntimeError("supervised trainer actual image digest is invalid")
    _require_expected_identity(
        evidence,
        expected_generation=generation,
        expected_git_commit=git_commit,
        expected_image_digest=resolved_image,
        expected_source_tree_digest=source_tree_digest,
        expected_lockfile_digest=lockfile_digest,
    )
    _require_valid_runtime_state(evidence)
    if evidence["state"] != "running":
        raise RuntimeError("supervised trainer did not enter the running state")
    if expectation_path is not None:
        if source_tree_digest is None or lockfile_digest is None:
            raise ValueError(
                "expectation persistence requires source and lockfile digests"
            )
        _write_json(
            expectation_path,
            {
                "generation": generation,
                "git_commit": git_commit,
                "image_digest": resolved_image,
                "lockfile_digest": lockfile_digest,
                "schema_version": "full_run_supervisor_expectation_v1",
                "source_tree_digest": source_tree_digest,
            },
        )
    return evidence


def supervised_run_status(
    *,
    expected_generation: str | None = None,
    expected_git_commit: str | None = None,
    expected_image_digest: str | None = None,
    expected_source_tree_digest: str | None = None,
    expected_lockfile_digest: str | None = None,
    expectation_path: Path | None = None,
    heartbeat_max_age_seconds: float = 120.0,
    max_runtime_seconds: float = 72.0 * 60.0 * 60.0,
    now: Now = _utc_now,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Inspect one retained container and fail closed for an expected run."""

    (
        expected_generation,
        expected_git_commit,
        expected_image_digest,
        expected_source_tree_digest,
        expected_lockfile_digest,
    ) = _resolve_expectations(
        expectation_path=expectation_path,
        expected_generation=expected_generation,
        expected_git_commit=expected_git_commit,
        expected_image_digest=expected_image_digest,
        expected_source_tree_digest=expected_source_tree_digest,
        expected_lockfile_digest=expected_lockfile_digest,
    )
    if expected_generation is not None and _NAME.fullmatch(expected_generation) is None:
        raise ValueError("expected_generation is invalid")
    if (
        expected_git_commit is not None
        and _GIT_SHA.fullmatch(expected_git_commit) is None
    ):
        raise ValueError("expected_git_commit is invalid")
    for field, value in (
        ("expected_image_digest", expected_image_digest),
        ("expected_source_tree_digest", expected_source_tree_digest),
        ("expected_lockfile_digest", expected_lockfile_digest),
    ):
        _validate_sha256(value, field=field)
    if heartbeat_max_age_seconds <= 0.0 or max_runtime_seconds <= 0.0:
        raise ValueError("heartbeat and runtime limits must be positive")
    names = _supervised_names(runner)
    if not names:
        if expected_generation is not None or expected_git_commit is not None:
            raise RuntimeError("expected supervised trainer is absent")
        return {
            "observed_at": _utc_now().isoformat(),
            "project": "trade-rl",
            "schema_version": "full_run_supervisor_status_v2",
            "state": "absent",
            "supervised": False,
        }
    if len(names) != 1:
        raise RuntimeError(
            "multiple supervised full trainers exist: " + ", ".join(names)
        )
    evidence = _inspect(names[0], runner)
    _require_expected_identity(
        evidence,
        expected_generation=expected_generation,
        expected_git_commit=expected_git_commit,
        expected_image_digest=expected_image_digest,
        expected_source_tree_digest=expected_source_tree_digest,
        expected_lockfile_digest=expected_lockfile_digest,
    )
    _require_valid_runtime_state(evidence)
    observed_now = now().astimezone(UTC)
    if evidence.get("state") == "running":
        generation = str(evidence.get("generation", ""))
        heartbeat = _read_heartbeat(names[0], generation=generation, runner=runner)
        heartbeat_time = _parse_utc(
            heartbeat.get("observed_at"), field="heartbeat.observed_at"
        )
        age = (observed_now - heartbeat_time).total_seconds()
        if age < -5.0:
            raise RuntimeError("supervised trainer heartbeat is in the future")
        if age > heartbeat_max_age_seconds:
            raise RuntimeError("supervised trainer heartbeat is stale")
        if heartbeat.get("git_commit") != evidence.get("git_commit"):
            raise RuntimeError("supervised trainer heartbeat git commit mismatch")
        if (
            expected_image_digest is not None
            and heartbeat.get("image_digest") != expected_image_digest
        ):
            raise RuntimeError("supervised trainer heartbeat image digest mismatch")
        started_at = _parse_utc(
            evidence.get("started_at"), field="container.started_at"
        )
        runtime = (observed_now - started_at).total_seconds()
        if runtime < -5.0:
            raise RuntimeError("supervised trainer start time is in the future")
        if runtime > max_runtime_seconds:
            raise RuntimeError("supervised trainer exceeded maximum runtime")
        evidence = {
            **evidence,
            "heartbeat": heartbeat,
            "heartbeat_age_seconds": age,
            "runtime_seconds": runtime,
        }
    return evidence


def stop_supervised_run(
    *,
    remove: bool = False,
    expected_generation: str | None = None,
    expected_git_commit: str | None = None,
    expected_image_digest: str | None = None,
    expected_source_tree_digest: str | None = None,
    expected_lockfile_digest: str | None = None,
    expectation_path: Path | None = None,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Capture status and logs before optionally removing the exact trainer."""

    (
        expected_generation,
        expected_git_commit,
        expected_image_digest,
        expected_source_tree_digest,
        expected_lockfile_digest,
    ) = _resolve_expectations(
        expectation_path=expectation_path,
        expected_generation=expected_generation,
        expected_git_commit=expected_git_commit,
        expected_image_digest=expected_image_digest,
        expected_source_tree_digest=expected_source_tree_digest,
        expected_lockfile_digest=expected_lockfile_digest,
    )
    names = _supervised_names(runner)
    if not names:
        if expected_generation is not None or expected_git_commit is not None:
            raise RuntimeError("expected supervised trainer is absent")
        return supervised_run_status(runner=runner)
    if len(names) != 1:
        raise RuntimeError(
            "multiple supervised full trainers exist: " + ", ".join(names)
        )
    name = names[0]
    inspect_before = _inspect(name, runner)
    _require_expected_identity(
        inspect_before,
        expected_generation=expected_generation,
        expected_git_commit=expected_git_commit,
        expected_image_digest=expected_image_digest,
        expected_source_tree_digest=expected_source_tree_digest,
        expected_lockfile_digest=expected_lockfile_digest,
    )
    if inspect_before["state"] == "running":
        command = ("docker", "stop", "--time", "60", name)
        _require_success(runner(command), command=command)
    evidence = _inspect(name, runner)
    log_tail = _container_logs(name, runner)
    evidence = {**evidence, "container_log_tail": log_tail}
    if remove:
        remove_command = ("docker", "rm", name)
        _require_success(runner(remove_command), command=remove_command)
        evidence = {**evidence, "removed": True}
        if expectation_path is not None:
            expectation_path.unlink(missing_ok=True)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("start", "status", "stop"))
    parser.add_argument("--generation")
    parser.add_argument("--container-name")
    parser.add_argument("--git-commit")
    parser.add_argument("--source-tree-digest")
    parser.add_argument("--lockfile-digest")
    parser.add_argument("--expected-generation")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-image-digest")
    parser.add_argument("--expected-source-tree-digest")
    parser.add_argument("--expected-lockfile-digest")
    parser.add_argument("--expectation-path", type=Path)
    parser.add_argument("--heartbeat-max-age-seconds", type=float, default=120.0)
    parser.add_argument("--max-runtime-hours", type=float, default=72.0)
    parser.add_argument("--metadata-mode", default="frozen_snapshot")
    parser.add_argument(
        "--compose-file", type=Path, default=Path("compose.training.yaml")
    )
    parser.add_argument("--evidence-path", type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.operation == "start":
        if not args.generation or not args.container_name or not args.git_commit:
            parser.error(
                "start requires --generation, --container-name, and --git-commit"
            )
        payload = start_supervised_run(
            generation=args.generation,
            container_name=args.container_name,
            git_commit=args.git_commit,
            metadata_mode=args.metadata_mode,
            compose_file=args.compose_file,
            source_tree_digest=args.source_tree_digest,
            lockfile_digest=args.lockfile_digest,
            expected_image_digest=args.expected_image_digest,
            expectation_path=args.expectation_path,
        )
    elif args.operation == "status":
        payload = supervised_run_status(
            expected_generation=args.expected_generation,
            expected_git_commit=args.expected_git_commit,
            expected_image_digest=args.expected_image_digest,
            expected_source_tree_digest=args.expected_source_tree_digest,
            expected_lockfile_digest=args.expected_lockfile_digest,
            expectation_path=args.expectation_path,
            heartbeat_max_age_seconds=args.heartbeat_max_age_seconds,
            max_runtime_seconds=args.max_runtime_hours * 60.0 * 60.0,
        )
    else:
        payload = stop_supervised_run(
            remove=args.remove,
            expected_generation=args.expected_generation,
            expected_git_commit=args.expected_git_commit,
            expected_image_digest=args.expected_image_digest,
            expected_source_tree_digest=args.expected_source_tree_digest,
            expected_lockfile_digest=args.expected_lockfile_digest,
            expectation_path=args.expectation_path,
        )
    if args.evidence_path is not None:
        _write_json(args.evidence_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
