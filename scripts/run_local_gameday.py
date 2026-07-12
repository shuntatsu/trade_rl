"""Run the deterministic exchange-free local GameDay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mars_lite.serving.local_gameday import exit_code_for_summary, run_local_gameday


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    summary = run_local_gameday(args.work_dir)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return exit_code_for_summary(summary)


if __name__ == "__main__":
    raise SystemExit(main())
