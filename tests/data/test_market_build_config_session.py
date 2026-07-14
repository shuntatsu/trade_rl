from __future__ import annotations

import json
from pathlib import Path

from trade_rl.data.config import load_market_build_request


def test_market_build_request_accepts_session_calendar(tmp_path: Path) -> None:
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1d",
                "calendar_kind": "session_calendar",
                "session_periods_per_year": 252,
                "features": [{"name": "ret", "kind": "log_return"}],
                "instruments": [
                    {
                        "symbol": "SPY",
                        "listed_at": "2020-01-01T00:00:00Z",
                        "volume_unit": "quote_notional",
                        "contract_multiplier": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    request = load_market_build_request(path)

    assert request.config.calendar_kind == "session_calendar"
    assert request.config.session_periods_per_year == 252
