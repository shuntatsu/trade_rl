"""Control-plane CLI for immutable serving-bundle registry operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from mars_lite.serving.registry import ModelRegistry


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage immutable serving bundles")
    parser.add_argument("--registry-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("candidate_dir", type=Path)

    activate = subparsers.add_parser("activate")
    activate.add_argument("version")
    activate.add_argument("--evidence-identity", required=True)

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--target-version")

    subparsers.add_parser("list")
    subparsers.add_parser("show-active")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = ModelRegistry(args.registry_dir)
    try:
        if args.command == "register":
            bundle = registry.register(args.candidate_dir)
            _print_json(
                {"version": bundle.version, "bundle_digest": bundle.bundle_digest}
            )
        elif args.command == "activate":
            bundle = registry.activate(args.version, args.evidence_identity)
            _print_json(
                {"version": bundle.version, "bundle_digest": bundle.bundle_digest}
            )
        elif args.command == "rollback":
            bundle = registry.rollback(args.target_version)
            _print_json(
                {"version": bundle.version, "bundle_digest": bundle.bundle_digest}
            )
        elif args.command == "list":
            _print_json({"versions": registry.list_versions()})
        elif args.command == "show-active":
            _print_json(registry.get_active_record().to_dict())
        else:
            parser.error(f"unsupported command: {args.command}")
        return 0
    except (ValueError, KeyError, LookupError, OSError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
