"""Command-line entrypoint for the local Trade RL Studio API."""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from trade_rl.studio.api import create_app
from trade_rl.studio.settings import StudioSettings

Runner = Callable[..., Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl studio")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start the local Studio API")
    start.add_argument("--project-root", type=Path, default=Path.cwd())
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)
    start.add_argument("--allow-remote", action="store_true")
    start.add_argument("--log-level", default="info")
    return parser


def _loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.command != "start":
        raise ValueError("unsupported studio command")
    if not 1 <= args.port <= 65_535:
        raise ValueError("port must be between 1 and 65535")
    if not args.allow_remote and not _loopback_host(args.host):
        raise ValueError("remote binding requires --allow-remote")
    settings = StudioSettings.from_environment(args.project_root)
    app = create_app(settings)
    if runner is None:
        import uvicorn

        resolved_runner: Runner = uvicorn.run
    else:
        resolved_runner = runner
    resolved_runner(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0
