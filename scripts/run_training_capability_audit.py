#!/usr/bin/env python3
"""Execute short real-training probes for every maintained learning backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_rl.operations.training_capability_audit import (
    run_training_capability_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/training-capability-audit"),
    )
    args = parser.parse_args()
    report = run_training_capability_audit(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
