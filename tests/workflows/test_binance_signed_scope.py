from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.workflows.binance_metadata_modes import (
    BinanceHistoricalSignedScope,
    resolution_from_historical_signed,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
START = datetime(2024, 12, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)
ISSUED = datetime(2026, 7, 17, tzinfo=UTC)


def _scope(**overrides: object) -> BinanceHistoricalSignedScope:
    values: dict[str, object] = {
        "market": "usds-m",
        "symbols": SYMBOLS,
        "coverage_start": START,
        "coverage_end": END,
        "issued_at": ISSUED,
        "source_uri": "operator://signed-binance-rules",
        "payload_digest": "a" * 64,
    }
    values.update(overrides)
    return BinanceHistoricalSignedScope(**values)


def _metadata() -> dict[str, dict[str, str | float]]:
    return {
        symbol: {
            "listed_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            "tick_size": 0.1,
            "lot_size": 0.001,
            "minimum_notional": 5.0,
        }
        for symbol in SYMBOLS
    }


def _histories(
    *, tick_size: float = 0.1
) -> dict[str, tuple[InstrumentExecutionRule, ...]]:
    return {
        symbol: (
            InstrumentExecutionRule(
                effective_at=START,
                tick_size=tick_size,
                lot_size=0.001,
                minimum_notional=5.0,
            ),
        )
        for symbol in SYMBOLS
    }


def test_signed_history_requires_exact_scope() -> None:
    result = resolution_from_historical_signed(
        metadata=_metadata(),
        execution_rule_histories=_histories(),
        signed_scope=_scope(),
        start_time=START,
        end_time=END,
    )

    assert result.identity_evidence["market"] == "usds-m"
    assert result.identity_evidence["symbols"] == SYMBOLS
    assert result.identity_evidence["point_in_time"] is True


def test_signed_scope_rejects_non_usdm_market() -> None:
    with pytest.raises(ValueError, match="market"):
        _scope(market="spot")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"symbols": tuple(reversed(SYMBOLS))}, "symbol"),
        ({"coverage_start": datetime(2024, 12, 2, tzinfo=UTC)}, "coverage"),
        ({"coverage_end": datetime(2026, 6, 30, tzinfo=UTC)}, "coverage"),
    ],
)
def test_signed_history_rejects_scope_mismatch(
    overrides: dict[str, object],
    match: str,
) -> None:
    scope = _scope(**overrides)
    with pytest.raises(ValueError, match=match):
        resolution_from_historical_signed(
            metadata=_metadata(),
            execution_rule_histories=_histories(),
            signed_scope=scope,
            start_time=START,
            end_time=END,
        )


def test_signed_history_rejects_zero_execution_rules() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        resolution_from_historical_signed(
            metadata=_metadata(),
            execution_rule_histories=_histories(tick_size=0.0),
            signed_scope=_scope(),
            start_time=START,
            end_time=END,
        )


def test_signed_history_rejects_rule_after_coverage() -> None:
    histories = _histories()
    histories["BTCUSDT"] = (
        *histories["BTCUSDT"],
        InstrumentExecutionRule(
            effective_at=datetime(2026, 7, 2, tzinfo=UTC),
            tick_size=0.1,
            lot_size=0.001,
            minimum_notional=5.0,
        ),
    )

    with pytest.raises(ValueError, match="coverage"):
        resolution_from_historical_signed(
            metadata=_metadata(),
            execution_rule_histories=histories,
            signed_scope=_scope(),
            start_time=START,
            end_time=END,
        )
