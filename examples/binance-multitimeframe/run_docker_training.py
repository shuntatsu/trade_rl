#!/usr/bin/env python3
"""Run incremental market-data synchronization before Docker GPU training."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Callable
from pathlib import Path

CommandRunner = Callable[[tuple[str, ...]], int]


def build_commands(compose_file: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prefix = ("docker", "compose", "-f", str(compose_file), "run", "--rm")
    return (
        (*prefix, "market-data-sync"),
        (*prefix, "trainer"),
    )


def _run_command(command: tuple[str, ...]) -> int:
    return subprocess.run(command, check=False).returncode


def run_training(
    *,
    compose_file: Path,
    runner: CommandRunner = _run_command,
) -> int:
    sync_command, trainer_command = build_commands(compose_file)
    sync_exit = runner(sync_command)
    if sync_exit != 0:
        return sync_exit
    return runner(trainer_command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("docker/compose.training.yaml"),
    )
    args = parser.parse_args(argv)
    return run_training(compose_file=args.compose_file)


if __name__ == "__main__":
    raise SystemExit(main())
