#!/usr/bin/env python3
"""Research-only Causal Alpha V4 command entrypoint."""

from __future__ import annotations

from trade_rl.workflows.universal_causal_alpha_v4_runner import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
