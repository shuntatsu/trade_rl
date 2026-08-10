from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.stored_instrument_catalog import (
    StoredIndicatorArtifactEvidence,
    StoredIndicatorSourceInventory,
    StoredInstrumentCatalog,
    build_stored_instrument_catalog,
)
from trade_rl.workflows.universal_instrument_partition import (
    build_universal_instrument_partition,
    load_universal_instrument_partition,
    universal_split_counts,
    write_universal_instrument_partition,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2026, 7, 1, tzinfo=UTC)


def _catalog(count: int) -> StoredInstrumentCatalog:
    symbols = tuple(f"ASSET{index:02d}USDT" for index in range(count))
    artifacts = tuple(
        StoredIndicatorArtifactEvidence(
            symbol=symbol,
            timeframe=timeframe,
            row_count=100,
            feature_count=8,
            available_value_count=100,
            first_event_time_ms=1_609_459_200_000,
            last_event_time_ms=1_782_864_000_000,
            payload_schema=f"npz_native_indicator_v1:{content_digest(timeframe)}",
            payload_sha256=content_digest((symbol, timeframe)),
            payload_bytes=4096,
        )
        for symbol in symbols
        for timeframe in _TIMEFRAMES
    )
    source = StoredIndicatorSourceInventory(
        cache_id="verified-native-indicators-v1",
        source_manifest_digest=content_digest("source-manifest"),
        market="usds-m",
        symbols=symbols,
        start_time=_START,
        end_time=_END,
        feature_config_digest=content_digest("features"),
        required_timeframes=_TIMEFRAMES,
        artifacts=artifacts,
    )
    return build_stored_instrument_catalog(
        source,
        research_start=_START,
        research_end=_END,
        metadata_digests={
            symbol: content_digest({"metadata": symbol}) for symbol in symbols
        },
    )


def test_universal_split_counts_require_zero_shot_support() -> None:
    assert universal_split_counts(15) == (9, 3, 3)
    assert universal_split_counts(20) == (12, 4, 4)
    with pytest.raises(ValueError, match="at least 15"):
        universal_split_counts(14)


def test_partition_is_catalog_bound_disjoint_and_deterministic() -> None:
    catalog = _catalog(15)

    first = build_universal_instrument_partition(catalog, seed=17)
    second = build_universal_instrument_partition(catalog, seed=17)

    assert first == second
    assert first.catalog_digest == catalog.digest
    assert first.split_counts == {"train": 9, "validation": 3, "test": 3}
    assert set(first.train_symbols).isdisjoint(first.validation_symbols)
    assert set(first.train_symbols).isdisjoint(first.test_symbols)
    assert set(first.validation_symbols).isdisjoint(first.test_symbols)
    assert set(first.train_symbols) | set(first.validation_symbols) | set(
        first.test_symbols
    ) == set(catalog.eligible_symbols)
    assert first.symbol_disjoint_manifest.digest == (
        first.symbol_disjoint_manifest_digest
    )


def test_partition_accessors_fail_closed_across_splits() -> None:
    partition = build_universal_instrument_partition(_catalog(15), seed=5)
    train = partition.train_symbols[0]
    validation = partition.validation_symbols[0]
    test = partition.test_symbols[0]

    assert partition.require_symbol(train, split="train") == train
    assert (
        partition.require_symbols(partition.validation_symbols, split="validation")
        == partition.validation_symbols
    )
    assert partition.require_symbol(test, split="test") == test

    with pytest.raises(ValueError, match="not declared for train"):
        partition.require_symbol(validation, split="train")
    with pytest.raises(ValueError, match="not declared for validation"):
        partition.require_symbol(train, split="validation")
    with pytest.raises(ValueError, match="not declared for test"):
        partition.require_symbol(validation, split="test")


def test_partition_json_round_trip_requires_matching_catalog_and_manifest(
    tmp_path: Path,
) -> None:
    catalog = _catalog(20)
    partition = build_universal_instrument_partition(catalog, seed=23)
    path = write_universal_instrument_partition(
        tmp_path / "universal-instrument-partition.json",
        partition,
    )

    assert (
        load_universal_instrument_partition(
            path,
            catalog=catalog,
            symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        )
        == partition
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["catalog_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog|digest"):
        load_universal_instrument_partition(
            path,
            catalog=catalog,
            symbol_disjoint_manifest=partition.symbol_disjoint_manifest,
        )
