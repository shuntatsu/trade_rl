from __future__ import annotations

import base64
import json
from pathlib import Path

_PATHS = (
    "trade_rl/workflows/market_walk_forward_config.py",
    "trade_rl/workflows/market_walk_forward.py",
    "examples/binance/walk-forward-smoke.json",
    "examples/binance-multitimeframe/walk-forward-full.json",
    "examples/binance-multitimeframe/walk-forward-growth-optimal.json",
    "tests/workflows/test_explicit_sealed_ledger_mode.py",
)
_PAYLOAD = {
    path: base64.b64encode(Path(path).read_bytes()).decode("ascii") for path in _PATHS
}
print(  # noqa: T201
    "PR198_SOURCE_EXPORT=" + base64.b64encode(json.dumps(_PAYLOAD).encode()).decode()
)
raise RuntimeError("intentional PR198 source export")
