from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.stored_instrument_catalog import (
    StoredIndicatorArtifactEvidence,
    StoredIndicatorSourceInventory,
    build_stored_instrument_catalog,
    load_stored_instrument_catalog,
    write_stored_instrument_catalog,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2026, 7, 1, tzinfo=UTC)


def _symbols(count: int) -> tuple[str, ...]:
    return tuple(f"ASSET{index:02d}USDT" for index in range(count))


def _artifact(
    symbol: str,
    timeframe: str,
    *,
    available_value_count: int = 100,
) -> StoredIndicatorArtifactEvidence:
    return StoredIndicatorArtifactEvidence(
        symbol=symbol,
        timeframe=timeframe,
        row_count=100,
        feature_count=8,
        available_value_count=available_value_count,
        first_event_time_ms=1_609_459_200_000,
        last_event_time_ms=1_782_864_000_000,
        payload_schema=f"npz_native_indicator_v1:{content_digest(timeframe)}",
        payload_sha256=content_digest({"symbol": symbol, "timeframe": timeframe}),
        payload_bytes=4096,
    )


def _source(
    count: int = 15,
    *,
    zero_available: tuple[str, str] | None = None,
    drop: tuple[str, str] | None = None,
) -> StoredIndicatorSourceInventory:
    symbols = _symbols(count)
    artifacts = []
    for symbol in symbols:
        for timeframe in _TIMEFRAMES:
            if drop == (symbol, timeframe):
                continue
            artifacts.append(
                _artifact(
                    symbol,
                    timeframe,
                    available_value_count=(
                        0 if zero_available == (symbol, timeframe) else 100
                    ),
                )
            )
    return StoredIndicatorSourceInventory(
        cache_id="verified-native-indicators-v1",
        source_manifest_digest=content_digest("source-manifest"),
        market="usds-m",
        symbols=symbols,
        start_time=_START,
        end_time=_END,
        feature_config_digest=content_digest("features"),
        required_timeframes=_TIMEFRAMES,
        artifacts=tuple(artifacts),
    )


def _metadata(symbols: tuple[str, ...]) -> dict[str, str]:
    return {symbol: content_digest({"metadata": symbol}) for symbol in symbols}


def test_catalog_records_eligibility_exclusions_and_source_evidence() -> None:
    source = _source(zero_available=("ASSET01USDT", "4h"))
    metadata = _metadata(source.symbols)
    metadata.pop("ASSET00USDT")

    catalog = build_stored_instrument_catalog(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=metadata,
    )

    assert catalog.source_cache_id == source.cache_id
    assert catalog.source_manifest_digest == source.source_manifest_digest
    assert catalog.market == "usds-m"
    assert catalog.feature_config_digest == source.feature_config_digest
    assert catalog.required_timeframes == _TIMEFRAMES
    assert catalog.research_start == _START
    assert catalog.research_end == _END
    assert catalog.eligible_symbols == source.symbols[2:]
    assert tuple(
        (item.symbol, item.reasons) for item in catalog.excluded_symbols
    ) == (
        ("ASSET00USDT", ("missing_execution_metadata",)),
        ("ASSET01USDT", ("no_available_values:4h",)),
    )
    assert tuple(symbol for symbol, _ in catalog.per_symbol_artifact_digests) == (
        source.symbols
    )
    assert dict(catalog.per_symbol_metadata_digests) == metadata
    assert len(catalog.digest) == 64


def test_catalog_json_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    source = _source()
    catalog = build_stored_instrument_catalog(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests=_metadata(source.symbols),
    )
    path = write_stored_instrument_catalog(tmp_path / "stored-instruments.json", catalog)

    assert load_stored_instrument_catalog(path) == catalog

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eligible_symbols"] = payload["eligible_symbols"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|closure"):
        load_stored_instrument_catalog(path)


def test_source_inventory_rejects_incomplete_symbol_timeframe_closure() -> None:
    with pytest.raises(ValueError, match="artifact closure"):
        _source(drop=("ASSET00USDT", "1d"))


def test_catalog_rejects_unknown_metadata_and_out_of_range_research() -> None:
    source = _source()
    metadata = _metadata(source.symbols)
    metadata["UNKNOWNUSDT"] = content_digest("unknown")
    with pytest.raises(ValueError, match="unknown symbols"):
        build_stored_instrument_catalog(
            source,
            research_start=_START,
            research_end=_END,
            metadata_digests=metadata,
        )

    with pytest.raises(ValueError, match="research interval"):
        build_stored_instrument_catalog(
            source,
            research_start=_START - timedelta(days=1),
            research_end=_END,
            metadata_digests=_metadata(source.symbols),
        )
