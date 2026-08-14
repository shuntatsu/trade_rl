"""Emit non-promotable diagnostics for one historical causal-alpha checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trade_rl.workflows.causal_alpha_research_diagnostics import (
    build_causal_alpha_research_report,
    load_causal_alpha_diagnostic_checkpoint_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a historical causal-alpha v2 selection checkpoint without "
            "making it resume- or promotion-eligible."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    snapshot = load_causal_alpha_diagnostic_checkpoint_v2(arguments.checkpoint)
    report = build_causal_alpha_research_report(snapshot)
    payload = (
        json.dumps(
            report.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if arguments.output is None:
        print(payload, end="")
    else:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
