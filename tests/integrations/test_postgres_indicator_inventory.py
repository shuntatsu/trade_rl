"""Contract tests for the metadata-only PostgreSQL indicator inventory."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.stored_instrument_catalog import (
    StoredIndicatorSourceInventory,
)
from trade_rl.integrations.postgres_indicator_artifacts import INDICATOR_CACHE_ID
from trade_rl.integrations.postgres_indicator_inventory import (
    load_postgres_indicator_source_inventory,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        self.database.queries.append((query, params))
        self.rows = (
            [self.database.manifest]
            if "indicator_manifests" in query
            else list(self.database.artifacts)
        )

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self.rows else self.rows[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class FakeDatabase:
    def __init__(self, symbol_count: int = 15) -> None:
        self.queries: list[tuple[str, object]] = []
        symbols = [f"ASSET{index:02d}USDT" for index in range(symbol_count)]
        feature_specs = {
            "base_timeframe": "15m",
            "feature_timeframes": ["1h", "4h", "1d"],
            "features": [
                {"name": "15m__return", "kind": "log_return"},
                {
                    "name": "1h__return",
                    "kind": "log_return",
                    "timeframe": "1h",
                },
                {
                    "name": "4h__return",
                    "kind": "log_return",
                    "timeframe": "4h",
                },
                {
                    "name": "1d__return",
                    "kind": "log_return",
                    "timeframe": "1d",
                },
            ],
        }
        self.manifest: tuple[object, ...] = (
            INDICATOR_CACHE_ID,
            "native_indicator_cache_v1",
            "usds-m",
            symbols,
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
            content_digest(feature_specs),
            feature_specs,
            len(symbols) * len(_TIMEFRAMES),
        )
        rows: list[tuple[object, ...]] = []
        for symbol_index, symbol in enumerate(symbols):
            for timeframe_index, timeframe in enumerate(_TIMEFRAMES):
                rows.append(
                    (
                        symbol,
                        timeframe,
                        100 + symbol_index,
                        1,
                        90 + timeframe_index,
                        1_609_459_200_000,
                        1_782_864_000_000,
                        f"npz_native_indicator_v1:{timeframe_index + 1:064x}",
                        content_digest({"symbol": symbol, "timeframe": timeframe}),
                        4096 + timeframe_index,
                    )
                )
        self.artifacts = tuple(reversed(rows))

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


def test_loads_catalog_source_inventory_without_npz_payload() -> None:
    database = FakeDatabase()

    inventory = load_postgres_indicator_source_inventory(database)

    assert isinstance(inventory, StoredIndicatorSourceInventory)
    assert inventory.cache_id == INDICATOR_CACHE_ID
    assert inventory.market == "usds-m"
    assert inventory.symbols == tuple(f"ASSET{index:02d}USDT" for index in range(15))
    assert inventory.required_timeframes == _TIMEFRAMES
    assert len(inventory.source_manifest_digest) == 64
    assert tuple(
        (item.symbol, item.timeframe) for item in inventory.artifacts
    ) == tuple(
        (symbol, timeframe) for symbol in inventory.symbols for timeframe in _TIMEFRAMES
    )
    assert inventory.artifact_for("ASSET03USDT", "4h").payload_bytes == 4098

    manifest_query, manifest_params = database.queries[0]
    artifact_query, artifact_params = database.queries[1]
    assert "indicator_manifests" in manifest_query
    assert manifest_params == (INDICATOR_CACHE_ID,)
    assert "indicator_artifacts" in artifact_query
    assert "npz_payload" not in artifact_query
    assert "payload_bytes" in artifact_query
    assert "ORDER BY symbol, timeframe" in artifact_query
    assert artifact_params == (INDICATOR_CACHE_ID,)


def test_source_inventory_identity_is_independent_of_database_row_order() -> None:
    database = FakeDatabase()
    first = load_postgres_indicator_source_inventory(database)
    database.queries.clear()
    database.artifacts = tuple(reversed(database.artifacts))

    second = load_postgres_indicator_source_inventory(database)

    assert second == first
    assert second.source_manifest_digest == first.source_manifest_digest


def test_rejects_missing_or_duplicate_artifact_metadata() -> None:
    missing = FakeDatabase()
    missing.artifacts = missing.artifacts[1:]
    with pytest.raises(FileNotFoundError, match="artifact metadata set mismatch"):
        load_postgres_indicator_source_inventory(missing)

    duplicate = FakeDatabase()
    duplicate.artifacts = (*duplicate.artifacts, duplicate.artifacts[0])
    with pytest.raises(ValueError, match="duplicate indicator artifact metadata"):
        load_postgres_indicator_source_inventory(duplicate)


def test_rejects_manifest_digest_and_feature_count_mismatch() -> None:
    manifest_tamper = FakeDatabase()
    manifest_row = list(manifest_tamper.manifest)
    manifest_row[6] = "0" * 64
    manifest_tamper.manifest = tuple(manifest_row)
    with pytest.raises(ValueError, match="feature config digest mismatch"):
        load_postgres_indicator_source_inventory(manifest_tamper)

    feature_count_tamper = FakeDatabase()
    rows = list(feature_count_tamper.artifacts)
    artifact_row = list(rows[0])
    artifact_row[3] = 2
    rows[0] = tuple(artifact_row)
    feature_count_tamper.artifacts = tuple(rows)
    with pytest.raises(ValueError, match="feature count mismatch"):
        load_postgres_indicator_source_inventory(feature_count_tamper)


def test_rejects_wrong_row_shape_or_scalar_types() -> None:
    manifest = FakeDatabase()
    manifest.manifest = manifest.manifest[:-1]
    with pytest.raises(ValueError, match="manifest row contract"):
        load_postgres_indicator_source_inventory(manifest)

    artifact_shape = FakeDatabase()
    artifact_shape.artifacts = (
        artifact_shape.artifacts[0][:-1],
        *artifact_shape.artifacts[1:],
    )
    with pytest.raises(ValueError, match="artifact metadata row contract"):
        load_postgres_indicator_source_inventory(artifact_shape)

    artifact_type = FakeDatabase()
    rows = list(artifact_type.artifacts)
    artifact_row = list(rows[0])
    artifact_row[2] = "100"
    rows[0] = tuple(artifact_row)
    artifact_type.artifacts = tuple(rows)
    with pytest.raises(ValueError, match="row_count must be an integer"):
        load_postgres_indicator_source_inventory(artifact_type)
