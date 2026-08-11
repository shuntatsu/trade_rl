from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.binance import BinanceMarket
from trade_rl.integrations.frozen_binance_metadata import (
    FrozenBinanceExchangeInfoTransport,
)


def exchange_info_fixture() -> dict[str, object]:
    return {
        "serverTime": 1_731_456_000_000,
        "symbols": [
            {"symbol": symbol, "status": "TRADING"}
            for symbol in ("ADAUSDT", "BTCUSDT", "ETHUSDT")
        ],
    }


def write_frozen_cache(root: Path, *, raw: bytes, market: str = "usds-m") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "exchange-info.raw.json").write_bytes(raw)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "market": market,
                "raw_payload_sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": datetime(2024, 11, 13, tzinfo=UTC).isoformat(),
                "schema_version": "frozen_metadata_cache_v1",
                "source_uri": "https://fapi.binance.com/fapi/v1/exchangeInfo",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_frozen_transport_loads_verified_snapshot_without_network(
    tmp_path: Path,
) -> None:
    raw = json.dumps(exchange_info_fixture(), sort_keys=True).encode()
    write_frozen_cache(tmp_path, raw=raw)

    snapshot = FrozenBinanceExchangeInfoTransport(
        tmp_path
    ).load_exchange_information_snapshot(market=BinanceMarket.USDS_M)

    assert snapshot.raw_payload == raw
    assert snapshot.raw_payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert snapshot.retrieved_at == datetime(2024, 11, 13, tzinfo=UTC)


@pytest.mark.parametrize(
    "mutation", ("missing_manifest", "bad_digest", "wrong_market")
)
def test_frozen_transport_fails_closed_on_incomplete_or_drifted_cache(
    tmp_path: Path, mutation: str
) -> None:
    raw = json.dumps(exchange_info_fixture(), sort_keys=True).encode()
    write_frozen_cache(
        tmp_path,
        raw=raw,
        market="coin-m" if mutation == "wrong_market" else "usds-m",
    )
    if mutation == "missing_manifest":
        (tmp_path / "manifest.json").unlink()
    elif mutation == "bad_digest":
        (tmp_path / "exchange-info.raw.json").write_bytes(raw + b" ")

    with pytest.raises((RuntimeError, ValueError)):
        FrozenBinanceExchangeInfoTransport(
            tmp_path
        ).load_exchange_information_snapshot(market=BinanceMarket.USDS_M)
