from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.postgres_indicator_artifacts import (
    INDICATOR_CACHE_ID,
    load_postgres_indicator_artifacts,
)
from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_TABLES,
)


def _payload(offset: int) -> bytes:
    event_time_ms = np.asarray([1000, 2000, 3000], dtype=np.int64)
    values = np.asarray([[offset], [offset + 1], [offset + 2]], dtype=np.float32)
    available = np.asarray([[False], [True], [True]], dtype=np.bool_)
    output = io.BytesIO()
    np.savez(
        output,
        event_time_ms=event_time_ms,
        values=values,
        available=available,
    )
    return output.getvalue()


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.rows: list[tuple[object, ...]] = []
        self.queries: list[tuple[str, object]] = database.queries

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        self.queries.append((query, params))
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
    def __init__(self) -> None:
        self.queries: list[tuple[str, object]] = []
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        features = [
            {"name": "15m__return", "kind": "log_return"},
            {"name": "1h__return", "kind": "log_return", "timeframe": "1h"},
            {"name": "4h__return", "kind": "log_return", "timeframe": "4h"},
            {"name": "1d__return", "kind": "log_return", "timeframe": "1d"},
        ]
        feature_specs = {
            "base_timeframe": "15m",
            "feature_timeframes": ["1h", "4h", "1d"],
            "features": features,
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
            len(symbols) * 4,
        )
        rows: list[tuple[object, ...]] = []
        for symbol_index, symbol in enumerate(symbols[1:]):
            for timeframe_index, timeframe in enumerate(("15m", "1h", "4h", "1d")):
                payload = _payload(symbol_index * 10 + timeframe_index)
                rows.append(
                    (
                        symbol,
                        timeframe,
                        3,
                        1,
                        2,
                        1000,
                        3000,
                        f"npz_native_indicator_v1:{timeframe_index + 1:064x}",
                        hashlib.sha256(payload).hexdigest(),
                        len(payload),
                        memoryview(payload),
                    )
                )
        self.artifacts = tuple(reversed(rows))

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)


def test_loads_requested_subset_in_declared_order_without_requiring_btc() -> None:
    database = FakeDatabase()
    symbols = ("SOLUSDT", "ETHUSDT", "BNBUSDT")
    timeframes = ("4h", "15m", "1d", "1h")

    bundle = load_postgres_indicator_artifacts(
        database,
        symbols=symbols,
        timeframes=timeframes,
    )

    assert bundle.symbols == symbols
    assert bundle.timeframes == timeframes
    assert tuple(
        (artifact.symbol, artifact.timeframe) for artifact in bundle.artifacts
    ) == tuple((symbol, timeframe) for symbol in symbols for timeframe in timeframes)
    assert tuple(bundle.by_symbol()) == symbols
    artifact = bundle.get("SOLUSDT", "4h")
    assert artifact.feature_names == ("4h__return",)
    assert artifact.values[:, 0].tolist() == pytest.approx([22.0, 23.0, 24.0])
    assert artifact.available.flags.writeable is False
    artifact_query, params = database.queries[1]
    assert "ORDER BY symbol, timeframe" in artifact_query
    assert params == (INDICATOR_CACHE_ID, list(symbols), list(timeframes))


def test_explicit_table_set_routes_indicator_queries() -> None:
    database = FakeDatabase()

    load_postgres_indicator_artifacts(
        database,
        symbols=("ETHUSDT", "BNBUSDT", "SOLUSDT"),
        timeframes=("15m", "1h", "4h", "1d"),
        tables=UNIVERSAL_202411_202607_TABLES,
    )

    assert UNIVERSAL_202411_202607_TABLES.indicator_manifest in database.queries[0][0]
    assert UNIVERSAL_202411_202607_TABLES.indicator_artifact in database.queries[1][0]


def test_rejects_missing_requested_artifact() -> None:
    database = FakeDatabase()
    database.artifacts = database.artifacts[1:]

    with pytest.raises(FileNotFoundError, match="artifact set mismatch"):
        load_postgres_indicator_artifacts(
            database,
            symbols=("ETHUSDT", "BNBUSDT", "SOLUSDT"),
            timeframes=("15m", "1h", "4h", "1d"),
        )


def test_rejects_tampered_npz_payload() -> None:
    database = FakeDatabase()
    rows = list(database.artifacts)
    row = list(rows[0])
    row[10] = memoryview(bytes(row[10]) + b"tampered")
    row[9] = len(row[10])
    rows[0] = tuple(row)
    database.artifacts = tuple(rows)

    with pytest.raises(ValueError, match="payload digest mismatch"):
        load_postgres_indicator_artifacts(
            database,
            symbols=("ETHUSDT", "BNBUSDT", "SOLUSDT"),
            timeframes=("15m", "1h", "4h", "1d"),
        )


def test_rejects_duplicate_or_unknown_request_identity() -> None:
    database = FakeDatabase()
    with pytest.raises(ValueError, match="symbols must be unique"):
        load_postgres_indicator_artifacts(
            database,
            symbols=("ETHUSDT", "ETHUSDT"),
            timeframes=("15m",),
        )
    with pytest.raises(ValueError, match="not in manifest"):
        load_postgres_indicator_artifacts(
            database,
            symbols=("XRPUSDT",),
            timeframes=("15m",),
        )
