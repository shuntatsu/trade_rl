from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
)
from trade_rl.workflows.binance_metadata_modes import (
    BinanceHistoricalSignedScope,
    BinanceMetadataMode,
    BinanceMetadataResolutionProvider,
    resolution_from_historical_signed,
    resolve_conservative_static,
    resolve_frozen_snapshot,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
START = datetime(2024, 12, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)
RETRIEVED = datetime(2026, 7, 17, 3, 0, tzinfo=UTC)


def _symbol_payload(
    symbol: str,
    *,
    tick_size: str = "0.10",
    lot_size: str = "0.001",
    minimum_notional: str = "5",
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "contractStatus": "TRADING",
        "onboardDate": 1_600_000_000_000,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick_size},
            {"filterType": "LOT_SIZE", "stepSize": lot_size},
            {"filterType": "MIN_NOTIONAL", "notional": minimum_notional},
        ],
    }


def _snapshot(
    *, payload: dict[str, object] | None = None
) -> BinanceExchangeInfoSnapshot:
    resolved = payload or {"symbols": [_symbol_payload(symbol) for symbol in SYMBOLS]}
    raw = json.dumps(resolved, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return BinanceExchangeInfoSnapshot(
        payload=resolved,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=RETRIEVED,
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _SnapshotTransport:
    def __init__(self, snapshot: BinanceExchangeInfoSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def load_exchange_information_snapshot(
        self, **_: object
    ) -> BinanceExchangeInfoSnapshot:
        self.calls += 1
        return self.snapshot


def test_metadata_modes_are_explicit() -> None:
    assert {item.value for item in BinanceMetadataMode} == {
        "historical_signed",
        "frozen_snapshot",
        "conservative_static",
    }


def test_frozen_snapshot_is_disclosed_bound_and_persisted_exactly(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    transport = _SnapshotTransport(snapshot)

    resolution = resolve_frozen_snapshot(
        transport=transport,
        market=BinanceMarket.USDS_M,
        symbols=SYMBOLS,
        start_time=START,
        end_time=END,
    )
    resolution.write_artifacts(tmp_path)

    assert transport.calls == 1
    assert resolution.mode is BinanceMetadataMode.FROZEN_SNAPSHOT
    assert resolution.execution_rule_histories is None
    assert resolution.evidence_digest
    assert resolution.identity_evidence["mode"] == "frozen_snapshot"
    assert (
        resolution.identity_evidence["raw_payload_sha256"]
        == snapshot.raw_payload_sha256
    )
    assert resolution.identity_evidence["point_in_time"] is False
    assert resolution.identity_evidence["authentication"] == "none"
    assert resolution.identity_evidence["as_of"] == RETRIEVED.isoformat()
    assert resolution.identity_evidence["limitations"]
    assert resolution.metadata["BTCUSDT"]["tick_size"] == pytest.approx(0.10)
    assert resolution.metadata["ETHUSDT"]["lot_size"] == pytest.approx(0.001)
    assert resolution.metadata["BNBUSDT"]["minimum_notional"] == pytest.approx(5.0)
    assert (tmp_path / "exchange-info.raw.json").read_bytes() == snapshot.raw_payload
    report = json.loads((tmp_path / "exchange-info.json").read_text(encoding="utf-8"))
    assert report["mode"] == "frozen_snapshot"
    assert report["evidence_digest"] == resolution.evidence_digest
    assert report["source_uri"] == snapshot.source_uri
    assert report["point_in_time"] is False
    assert report["production_status"] == "NO-GO"


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {"symbols": [_symbol_payload("BTCUSDT"), _symbol_payload("ETHUSDT")]},
            "BNBUSDT",
        ),
        (
            {
                "symbols": [
                    _symbol_payload("BTCUSDT", tick_size="0"),
                    _symbol_payload("ETHUSDT"),
                    _symbol_payload("BNBUSDT"),
                ]
            },
            "tick_size",
        ),
        (
            {
                "symbols": [
                    {**_symbol_payload("BTCUSDT"), "filters": []},
                    _symbol_payload("ETHUSDT"),
                    _symbol_payload("BNBUSDT"),
                ]
            },
            "PRICE_FILTER",
        ),
    ],
)
def test_frozen_snapshot_fails_closed_for_missing_or_nonpositive_rules(
    payload: dict[str, object],
    match: str,
) -> None:
    transport = _SnapshotTransport(_snapshot(payload=payload))

    with pytest.raises(ValueError, match=match):
        resolve_frozen_snapshot(
            transport=transport,
            market=BinanceMarket.USDS_M,
            symbols=SYMBOLS,
            start_time=START,
            end_time=END,
        )


def test_resolution_provider_reuses_one_resolution_for_both_dataset_builds() -> None:
    calls = 0
    resolution = resolve_frozen_snapshot(
        transport=_SnapshotTransport(_snapshot()),
        market=BinanceMarket.USDS_M,
        symbols=SYMBOLS,
        start_time=START,
        end_time=END,
    )

    def resolver() -> object:
        nonlocal calls
        calls += 1
        return resolution

    provider = BinanceMetadataResolutionProvider(resolver)

    assert provider.get() is resolution
    assert provider.get() is resolution
    assert calls == 1


def test_historical_signed_resolution_preserves_effective_history() -> None:
    rule = InstrumentExecutionRule(
        effective_at=START,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
    )
    metadata = {
        symbol: {
            "listed_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            "tick_size": rule.tick_size,
            "lot_size": rule.lot_size,
            "minimum_notional": rule.minimum_notional,
        }
        for symbol in SYMBOLS
    }
    histories = {symbol: (rule,) for symbol in SYMBOLS}
    scope = BinanceHistoricalSignedScope(
        market=BinanceMarket.USDS_M.value,
        symbols=SYMBOLS,
        coverage_start=START,
        coverage_end=END,
        issued_at=RETRIEVED,
        source_uri="operator://signed-binance-rules",
        payload_digest="a" * 64,
    )

    resolution = resolution_from_historical_signed(
        metadata=metadata,
        execution_rule_histories=histories,
        signed_scope=scope,
        start_time=START,
        end_time=END,
    )

    assert resolution.mode is BinanceMetadataMode.HISTORICAL_SIGNED
    assert resolution.execution_rule_histories == histories
    assert resolution.identity_evidence["authentication"] == "hmac-sha256"
    assert resolution.identity_evidence["point_in_time"] is True
    assert resolution.identity_evidence["limitations"] == ()
    assert resolution.identity_evidence["source_uri"] == scope.source_uri


def test_conservative_static_requires_versioned_payload_and_positive_stress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conservative-static.json"
    payload = {
        "schema_version": "binance_conservative_static_v1",
        "as_of": RETRIEVED.isoformat(),
        "source_uri": "operator://conservative-static",
        "symbols": {
            symbol: {
                "listed_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
                "tick_size": 0.1,
                "lot_size": 0.001,
                "minimum_notional": 5.0,
            }
            for symbol in SYMBOLS
        },
        "stress_factors": {
            "tick_size": 2.0,
            "lot_size": 2.0,
            "minimum_notional": 5.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    resolution = resolve_conservative_static(
        path=path,
        symbols=SYMBOLS,
        start_time=START,
        end_time=END,
    )

    assert resolution.mode is BinanceMetadataMode.CONSERVATIVE_STATIC
    assert resolution.execution_rule_histories is None
    assert resolution.identity_evidence["stress_factors"] == payload["stress_factors"]
    assert resolution.identity_evidence["point_in_time"] is False
    assert resolution.identity_evidence["authentication"] == "operator-declared"

    payload["stress_factors"]["lot_size"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stress_factors.lot_size"):
        resolve_conservative_static(
            path=path,
            symbols=SYMBOLS,
            start_time=START,
            end_time=END,
        )


def test_evidence_artifacts_fail_closed_instead_of_overwriting(tmp_path: Path) -> None:
    resolution = resolve_frozen_snapshot(
        transport=_SnapshotTransport(_snapshot()),
        market=BinanceMarket.USDS_M,
        symbols=SYMBOLS,
        start_time=START,
        end_time=END,
    )
    resolution.write_artifacts(tmp_path)

    with pytest.raises(FileExistsError, match="exchange-info"):
        resolution.write_artifacts(tmp_path)
