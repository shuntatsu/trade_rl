from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    load_causal_alpha_v3_signal_forensics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze persisted Causal Alpha V3 Signal V2 artifacts without "
            "refitting or replaying the source run."
        )
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help="Causal Alpha V3 run root containing run-manifest.json and signal/records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path outside the source run root",
    )
    return parser


def _canonical_json(payload: object) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _require_external_output(run_root: Path, output: Path) -> None:
    source = run_root.resolve()
    destination = output.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("V3 signal forensics output must remain outside source run root")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = Path(args.run_root)
    output = None if args.output is None else Path(args.output)
    if output is not None:
        _require_external_output(run_root, output)

    report = load_causal_alpha_v3_signal_forensics(run_root)
    encoded = _canonical_json(report.to_payload())
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
