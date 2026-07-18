#!/usr/bin/env python3
"""Verify packaged identity and supervise one full-research phase."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from trade_rl.artifacts.provenance import capture_runtime_provenance

_ROOT = Path(__file__).resolve().parents[2]
_SAFE_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ChildProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self) -> int: ...

    def send_signal(self, signal_number: int) -> None: ...


PopenFactory = Callable[[Sequence[str]], ChildProcess]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    resolved = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return resolved.astimezone(UTC).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _optional_sha256(value: str | None, *, field: str) -> str | None:
    if value is None or value == "":
        return None
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def resolve_generation_root(runs_root: str | Path, generation: str) -> Path:
    """Resolve one safe generation below the configured runs root."""

    if _SAFE_GENERATION.fullmatch(generation) is None:
        raise ValueError("generation must be a safe non-empty identifier")
    root = Path(runs_root).resolve()
    resolved = (root / generation).resolve()
    if resolved.parent != root:
        raise ValueError("generation must resolve directly below the runs root")
    return resolved


def build_training_command(
    *,
    python_executable: str,
    repository_root: Path,
    work_root: Path,
    cache_root: Path,
    metadata_mode: str,
    phase: str,
    conservative_static_path: str | None = None,
    selection_authorization: str | None = None,
    selection_public_keys: str | None = None,
    confirmation: str | None = None,
    confirmation_public_keys: str | None = None,
    trusted_now: str | None = None,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(
            repository_root
            / "examples"
            / "binance-multitimeframe"
            / "run_full_research_hardened.py"
        ),
        "--phase",
        phase,
        "--work-root",
        str(work_root),
        "--cache-root",
        str(cache_root),
        "--metadata-mode",
        metadata_mode,
    ]
    optional = (
        ("--conservative-static-path", conservative_static_path),
        ("--selection-authorization", selection_authorization),
        ("--selection-public-keys", selection_public_keys),
        ("--confirmation", confirmation),
        ("--confirmation-public-keys", confirmation_public_keys),
        ("--trusted-now", trusted_now),
    )
    for flag, value in optional:
        if value:
            command.extend((flag, value))
    return tuple(command)


def _default_popen(command: Sequence[str]) -> ChildProcess:
    return subprocess.Popen(tuple(command))


def run_with_heartbeat(
    command: Sequence[str],
    *,
    heartbeat_path: Path,
    generation: str,
    git_commit: str,
    image_digest: str | None,
    phase: str,
    interval_seconds: float = 30.0,
    popen_factory: PopenFactory = _default_popen,
    sleeper: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utc_now,
) -> int:
    """Run one child while atomically publishing liveness and terminal evidence."""

    if _GIT_SHA.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be a lowercase 40-character SHA")
    resolved_image = _optional_sha256(image_digest, field="image_digest")
    if interval_seconds < 0.0:
        raise ValueError("interval_seconds must not be negative")
    process = popen_factory(tuple(command))
    previous_handlers: dict[int, Any] = {}

    def forward(signal_number: int, _frame: object) -> None:
        process.send_signal(signal_number)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, forward)
        except ValueError:
            # Signal handlers can only be installed from the main thread.
            previous_handlers.clear()
            break

    def payload(*, state: str, exit_code: int | None) -> dict[str, object]:
        return {
            "command": tuple(str(value) for value in command),
            "exit_code": exit_code,
            "generation": generation,
            "git_commit": git_commit,
            "image_digest": resolved_image,
            "observed_at": _iso(now()),
            "phase": phase,
            "pid": process.pid,
            "schema_version": "full_run_heartbeat_v1",
            "state": state,
        }

    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                _atomic_json(
                    heartbeat_path,
                    payload(
                        state="exited" if return_code == 0 else "failed",
                        exit_code=return_code,
                    ),
                )
                return int(return_code)
            _atomic_json(heartbeat_path, payload(state="running", exit_code=None))
            sleeper(interval_seconds)
    finally:
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


def _parse_dirty(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("TRADE_RL_GIT_DIRTY must be true or false")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/var/runs"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/workspace/var/cache/binance-vision")
    )
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    generation = _required_env("TRADE_RL_RUN_GENERATION")
    git_commit = _required_env("TRADE_RL_GIT_COMMIT").lower()
    git_dirty = _parse_dirty(_required_env("TRADE_RL_GIT_DIRTY"))
    phase = os.environ.get("TRADE_RL_RESEARCH_PHASE", "develop")
    metadata_mode = os.environ.get("TRADE_RL_METADATA_MODE", "frozen_snapshot")
    image_digest = _optional_sha256(
        os.environ.get("TRADE_RL_IMAGE_DIGEST"), field="TRADE_RL_IMAGE_DIGEST"
    )
    work_root = resolve_generation_root(args.runs_root, generation)
    work_root.mkdir(parents=True, exist_ok=True)

    provenance = capture_runtime_provenance(
        _ROOT,
        git_commit=git_commit,
        git_dirty=git_dirty,
        deterministic_seed_config={
            "generation": generation,
            "metadata_mode": metadata_mode,
            "phase": phase,
        },
        image_digest=image_digest,
    )
    _atomic_json(work_root / "entrypoint-provenance.json", asdict(provenance))

    preflight = (
        sys.executable,
        str(
            _ROOT / "examples" / "binance-multitimeframe" / "training_cuda_preflight.py"
        ),
        "--output",
        str(work_root / "cuda-preflight.json"),
    )
    preflight_code = run_with_heartbeat(
        preflight,
        heartbeat_path=work_root / "heartbeat.json",
        generation=generation,
        git_commit=git_commit,
        image_digest=image_digest,
        phase=f"{phase}:cuda-preflight",
        interval_seconds=args.heartbeat_interval,
    )
    if preflight_code != 0:
        return preflight_code

    command = build_training_command(
        python_executable=sys.executable,
        repository_root=_ROOT,
        work_root=work_root,
        cache_root=args.cache_root,
        metadata_mode=metadata_mode,
        phase=phase,
        conservative_static_path=os.environ.get("TRADE_RL_CONSERVATIVE_STATIC_PATH"),
        selection_authorization=os.environ.get("TRADE_RL_SELECTION_AUTHORIZATION"),
        selection_public_keys=os.environ.get("TRADE_RL_SELECTION_PUBLIC_KEYS"),
        confirmation=os.environ.get("TRADE_RL_CONFIRMATION_EVIDENCE"),
        confirmation_public_keys=os.environ.get("TRADE_RL_CONFIRMATION_PUBLIC_KEYS"),
        trusted_now=os.environ.get("TRADE_RL_TRUSTED_NOW"),
    )
    return run_with_heartbeat(
        command,
        heartbeat_path=work_root / "heartbeat.json",
        generation=generation,
        git_commit=git_commit,
        image_digest=image_digest,
        phase=phase,
        interval_seconds=args.heartbeat_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
