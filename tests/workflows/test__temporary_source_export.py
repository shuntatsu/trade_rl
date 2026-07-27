from __future__ import annotations

import base64
from pathlib import Path


def test_export_market_walk_forward_source() -> None:
    source = Path("trade_rl/workflows/market_walk_forward.py").read_bytes()
    encoded = base64.b64encode(source).decode("ascii")
    print(f"MARKET_WALK_FORWARD_SOURCE_B64={encoded}")
    raise AssertionError("temporary source export")
