from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from trade_rl.reporting.markdown import render_run_report_markdown
from trade_rl.reporting.run_report import build_run_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic fact-only report from persisted run artifacts."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("chat", "json"),
        default="chat",
    )
    parser.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    root = Path(args.root)
    if not root.is_dir():
        print(f"run report root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    report = build_run_report(root)
    content = (
        render_run_report_markdown(report)
        if args.profile == "chat"
        else report.to_json()
    )
    if args.output == "-":
        sys.stdout.write(content)
        return 0

    output = Path(args.output)
    output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
