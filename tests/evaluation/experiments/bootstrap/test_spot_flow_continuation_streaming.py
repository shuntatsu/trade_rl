from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade_rl.evaluation.experiments.bootstrap import (
    spot_flow_continuation_diagnostic as diagnostic,
)

QUARTER_HOUR_MS = 900_000


def _start(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def test_streaming_day_aggregation_matches_frozen_interval_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    date = "2021-01-15"
    start = _start(date)
    rows = (
        f"1,100,2,10,10,{start + 1},False,True\n"
        f"2,100,1,11,11,{start + QUARTER_HOUR_MS - 1},True,True\n"
        f"3,0,0,-1,-1,{start + QUARTER_HOUR_MS},True,False\n"
        f"4,300,1,12,12,{start + QUARTER_HOUR_MS},False,True\n"
        f"5,100,1,13,13,{start + 2 * QUARTER_HOUR_MS - 1},True,True\n"
    ).encode()

    parsed = diagnostic.parse_spot_aggtrades_csv(rows, expected_date=date)
    expected_first = diagnostic.aggregate_spot_interval(
        parsed, decision_time_ms=start + QUARTER_HOUR_MS
    )
    expected_second = diagnostic.aggregate_spot_interval(
        parsed, decision_time_ms=start + 2 * QUARTER_HOUR_MS
    )

    def materialization_forbidden(_: bytes) -> list[list[str]]:
        raise AssertionError("streaming aggregation must not materialize the full CSV")

    monkeypatch.setattr(diagnostic, "_decode_csv", materialization_forbidden)
    signals = diagnostic.aggregate_spot_day_csv(rows, expected_date=date)

    assert len(signals) == 96
    assert signals[0] == expected_first == pytest.approx(1.0 / 3.0)
    assert signals[1] == expected_second == pytest.approx(0.5)
    assert signals[2:] == (None,) * 94


def test_streaming_day_aggregation_rejects_partial_sentinel() -> None:
    date = "2021-01-15"
    start = _start(date)
    payload = f"1,0,1,-1,-1,{start + 1},True,False\n".encode()

    with pytest.raises(ValueError, match="sentinel|price|quantity|malformed"):
        diagnostic.aggregate_spot_day_csv(payload, expected_date=date)
