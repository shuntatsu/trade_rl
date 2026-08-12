from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.postgres_indicator_artifacts import INDICATOR_CACHE_ID
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.workflows.postgres_universal_instrument_artifacts import (
    materialize_postgres_universal_instrument_artifacts,
)
from trade_rl.workflows.universal_instrument_artifacts import (
    STORED_INSTRUMENTS_FILENAME,
    SYMBOL_DISJOINT_FILENAME,
    UNIVERSAL_INSTRUMENT_PARTITION_FILENAME,
    load_universal_instrument_artifact_bundle,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_START = datetime(2021, 1, 1, tzinfo=UTC)
_END = datetime(2026, 7, 1, tzinfo=UTC)


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
    def __init__(
        self,
        *,
        cache_id: str = INDICATOR_CACHE_ID,
        symbol_count: int = 15,
    ) -> None:
        self.queries: list[tuple[str, object]] = []
        self.symbols = tuple(f"ASSET{index:02d}USDT" for index in range(symbol_count))
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
            cache_id,
            "native_indicator_cache_v1",
            "usds-m",
            self.symbols,
            _START,
            _END,
            content_digest(feature_specs),
            feature_specs,
            len(self.symbols) * len(_TIMEFRAMES),
        )
        self.artifacts = tuple(
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
            for symbol_index, symbol in enumerate(self.symbols)
            for timeframe_index, timeframe in enumerate(_TIMEFRAMES)
        )

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


def _metadata(symbols: tuple[str, ...]) -> dict[str, str]:
    return {symbol: content_digest({"metadata": symbol}) for symbol in symbols}


def test_materializes_postgres_inventory_into_exact_bundle(tmp_path: Path) -> None:
    cache_id = "universal-instrument-source-v1"
    database = FakeDatabase(cache_id=cache_id)
    output = tmp_path / "universal-instruments"

    paths = materialize_postgres_universal_instrument_artifacts(
        database,
        output_dir=output,
        research_start=_START,
        research_end=_END,
        metadata_digests=_metadata(database.symbols),
        seed=17,
        cache_id=cache_id,
    )

    assert paths.root == output
    assert {path.name for path in output.iterdir()} == {
        STORED_INSTRUMENTS_FILENAME,
        SYMBOL_DISJOINT_FILENAME,
        UNIVERSAL_INSTRUMENT_PARTITION_FILENAME,
    }
    bundle = load_universal_instrument_artifact_bundle(output)
    assert bundle.catalog.source_cache_id == cache_id
    assert bundle.catalog.eligible_symbols == database.symbols
    assert bundle.partition.split_counts == {
        "train": 9,
        "validation": 3,
        "test": 3,
    }

    assert len(database.queries) == 2
    for query, params in database.queries:
        assert params == (cache_id,)
        assert "npz_payload" not in query


def test_explicit_table_set_routes_workflow_inventory_queries(tmp_path: Path) -> None:
    database = FakeDatabase()

    materialize_postgres_universal_instrument_artifacts(
        database,
        output_dir=tmp_path / "universal-instruments",
        research_start=_START,
        research_end=_END,
        metadata_digests=_metadata(database.symbols),
        seed=17,
        tables=UNIVERSAL_202411_202607_TABLES,
    )

    assert UNIVERSAL_202411_202607_TABLES.indicator_manifest in database.queries[0][0]
    assert UNIVERSAL_202411_202607_TABLES.indicator_artifact in database.queries[1][0]


def test_missing_execution_metadata_fails_before_publication(tmp_path: Path) -> None:
    database = FakeDatabase()
    metadata = _metadata(database.symbols)
    metadata.pop(database.symbols[0])
    output = tmp_path / "universal-instruments"

    with pytest.raises(ValueError, match="at least 15"):
        materialize_postgres_universal_instrument_artifacts(
            database,
            output_dir=output,
            research_start=_START,
            research_end=_END,
            metadata_digests=metadata,
            seed=17,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".universal-instruments.staging-*"))
