#!/usr/bin/env python3
"""Research-only Causal Alpha V6 command entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trade_rl.workflows.universal_causal_alpha_v6_runner import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
