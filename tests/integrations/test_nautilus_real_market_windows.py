from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.historical_subprocess import (
    run_historical_target_intervals_subprocess,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "nautilus"
    / "btcusdt-usdsm-representative-15m.json"
)
_BAR_SPAN_MS = 15 * 60 * 1000


def _payload() -> dict[str, Any]:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _source_bars(rows: list[list[object]]) -> tuple[SourceBar, ...]:
    bars: list[SourceBar] = []
    for row in rows:
        assert len(row) == 7
        open_time_ms, open_price, high_price, low_price, close_price, mark, index = row
        assert isinstance(open_time_ms, int)
        bars.append(
            SourceBar(
                open_ns=open_time_ms * 1_000_000,
                close_ns=(open_time_ms + _BAR_SPAN_MS - 1) * 1_000_000,
                open_price=float(open_price),
                high_price=float(high_price),
                low_price=float(low_price),
                close_price=float(close_price),
                mark_price=float(mark),
                index_price=float(index),
            )
        )
    return tuple(bars)


def _round_trip_intervals(rows: list[list[object]]) -> tuple[NautilusHistoricalTargetInterval, ...]:
    bars = _source_bars(rows)
    assert len(bars) == 16
    return (
        NautilusHistoricalTargetInterval(
            sequence=1,
            target_exposure=0.10,
            allocated_equity=100_000.0,
            source_bars=bars[:8],
        ),
        NautilusHistoricalTargetInterval(
            sequence=2,
            target_exposure=0.0,
            allocated_equity=100_000.0,
            source_bars=bars[8:],
        ),
    )


@pytest.mark.nautilus
def test_representative_fixture_is_time_selected_real_binance_data() -> None:
    payload = _payload()

    assert payload["schema_version"] == "btc_usdsm_representative_windows_v1"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "15m"
    assert payload["canonical_range"] == [
        "2021-01-01T00:00:00Z",
        "2026-07-01T00:00:00Z",
        192672,
    ]
    assert payload["selection"]["method"] == "time_quantiles_floor_15m"
    assert payload["selection"]["quantiles"] == [0.1, 0.5, 0.9]
    assert payload["selection"]["window_bars"] == 16
    assert [window["time_quantile"] for window in payload["windows"]] == [
        0.1,
        0.5,
        0.9,
    ]
    assert all(len(window["rows"]) == 16 for window in payload["windows"])

    for window in payload["windows"]:
        open_times = [row[0] for row in window["rows"]]
        assert all(isinstance(value, int) for value in open_times)
        assert all(
            right - left == _BAR_SPAN_MS
            for left, right in zip(open_times, open_times[1:], strict=True)
        )

    assert payload["windows"][2]["funding"] == [
        ["1765526400007", "0.00004698", "92392.37302174", "Regular"]
    ]


@pytest.mark.nautilus
def test_representative_real_windows_replay_deterministically_and_finish_flat() -> None:
    payload = _payload()

    for window in payload["windows"]:
        intervals = _round_trip_intervals(window["rows"])
        first = run_historical_target_intervals_subprocess(
            intervals,
            starting_balance=Decimal("100000"),
            no_trade_band=0.0,
        )
        second = run_historical_target_intervals_subprocess(
            intervals,
            starting_balance=Decimal("100000"),
            no_trade_band=0.0,
        )

        assert first.worker_pid != second.worker_pid
        assert first.execution == second.execution
        assert first.execution.runtime_version == "1.230.0"
        assert first.execution.fills
        assert first.execution.fills[0].position_lots > 0
        assert first.execution.fills[-1].position_lots == 0
        assert first.execution.terminal_position_lots == 0
        assert first.execution.terminal_open_orders == 0
