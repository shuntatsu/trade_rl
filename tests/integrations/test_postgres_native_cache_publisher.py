from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.integrations.postgres_native_cache_publisher import (
    publish_native_cache,
)
from trade_rl.integrations.postgres_universal_source import (
    RawSymbolSource,
    UniversalSourceScope,
)
from trade_rl.integrations.native_indicator_materializer import (
    NativeCacheBuild,
    build_native_indicator_cache,
)


class TransactionRecordingCursor:
    def __init__(self, connection: TransactionRecordingConnection) -> None:
        self.connection = connection
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> TransactionRecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        normalized = " ".join(query.split())
        self.connection.queries.append((normalized, params))
        if normalized.startswith("CREATE"):
            self.rows = []
            return
        assert isinstance(params, tuple)
        cache_id = str(params[0])
        if normalized.startswith("SELECT manifest_digest FROM"):
            manifest = self.connection.manifests.get(cache_id)
            self.rows = [] if manifest is None else [(manifest[0],)]
        elif (
            normalized.startswith("INSERT INTO") and "indicator_manifests" in normalized
        ):
            assert isinstance(params[7], str)
            assert isinstance(json.loads(params[7]), dict)
            self.connection.manifests[cache_id] = (str(params[-1]), int(params[8]))
            self.rows = []
        elif normalized.startswith("SELECT manifest_digest, artifact_count"):
            manifest = self.connection.manifests.get(cache_id)
            self.rows = [] if manifest is None else [manifest]
        elif normalized.startswith("SELECT symbol, timeframe, payload_sha256"):
            self.rows = [
                (symbol, timeframe, digest)
                for (stored_cache, symbol, timeframe), digest in sorted(
                    self.connection.artifacts.items()
                )
                if stored_cache == cache_id
            ]
        elif normalized.startswith("SELECT COUNT(*) FROM"):
            kind = "funding" if "funding_202411_202607" in normalized else "kline"
            self.rows = [(self.connection.row_counts[kind].get(cache_id, 0),)]
        else:  # pragma: no cover - proves the publisher's SQL surface stays explicit
            raise AssertionError(normalized)

    def executemany(self, query: str, params: list[tuple[object, ...]]) -> None:
        normalized = " ".join(query.split())
        self.connection.queries.append((normalized, tuple(params)))
        if "indicator_artifacts" in normalized:
            for row in params:
                self.connection.artifacts[(str(row[0]), str(row[1]), str(row[2]))] = (
                    str(row[9])
                )
        elif "funding_202411_202607" in normalized:
            for row in params:
                cache_id = str(row[0])
                self.connection.row_counts["funding"][cache_id] = (
                    self.connection.row_counts["funding"].get(cache_id, 0) + 1
                )
        elif "klines_202411_202607" in normalized:
            for row in params:
                cache_id = str(row[0])
                self.connection.row_counts["kline"][cache_id] = (
                    self.connection.row_counts["kline"].get(cache_id, 0) + 1
                )
        else:  # pragma: no cover - proves insert routing
            raise AssertionError(normalized)

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self.rows else self.rows[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class TransactionRecordingConnection:
    def __init__(self) -> None:
        self.manifests: dict[str, tuple[str, int]] = {}
        self.artifacts: dict[tuple[str, str, str], str] = {}
        self.row_counts: dict[str, dict[str, int]] = {"kline": {}, "funding": {}}
        self.queries: list[tuple[str, object]] = []
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> TransactionRecordingCursor:
        return TransactionRecordingCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


def native_build_fixture() -> NativeCacheBuild:
    count = 60
    start = datetime(2024, 11, 13, tzinfo=UTC)
    timestamps = np.datetime64(start.replace(tzinfo=None), "ns") + np.arange(
        count
    ) * np.timedelta64(1, "m")
    rows = np.arange(count, dtype=np.float64)
    open_ = 100.0 + rows
    source = RawSymbolSource(
        timestamps=timestamps,
        open=open_,
        high=open_ + 2.0,
        low=open_ - 1.0,
        close=open_ + 1.0,
        base_volume=np.ones(count, dtype=np.float64),
        funding_timestamps=np.asarray([timestamps[30]], dtype="datetime64[ns]"),
        funding_rate=np.asarray([0.0001], dtype=np.float64),
        derivative_timestamps=timestamps.copy(),
        derivative_values=np.zeros((count, 4), dtype=np.float64),
        orderflow_timestamps=timestamps.copy(),
        orderflow_values=np.zeros((count, 5), dtype=np.float64),
    )
    scope = UniversalSourceScope(
        symbols=("BTCUSDT",), start=start, end=start + timedelta(minutes=count)
    )
    return build_native_indicator_cache({"BTCUSDT": source}, scope=scope)


def mutated_build(build: NativeCacheBuild) -> NativeCacheBuild:
    artifact = build.artifacts[0]
    payload = artifact.payload + b"drift"
    changed = replace(
        artifact,
        payload=payload,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return replace(build, artifacts=(changed, *build.artifacts[1:]))


def test_publish_is_atomic_idempotent_and_rejects_drift() -> None:
    connection = TransactionRecordingConnection()
    build = native_build_fixture()

    first = publish_native_cache(
        connection, build, tables=UNIVERSAL_202411_202607_TABLES
    )
    second = publish_native_cache(
        connection, build, tables=UNIVERSAL_202411_202607_TABLES
    )

    assert first == second
    assert first.artifact_count == 4
    assert first.funding_row_count == 1
    assert connection.commit_count == 2
    with pytest.raises(FileExistsError, match="different content"):
        publish_native_cache(
            connection,
            mutated_build(build),
            tables=UNIVERSAL_202411_202607_TABLES,
        )
    assert connection.rollback_count == 1


def test_publish_rolls_back_any_insert_failure() -> None:
    connection = TransactionRecordingConnection()
    build = native_build_fixture()
    original = connection.cursor

    class FailingCursor(TransactionRecordingCursor):
        def executemany(self, query: str, params: list[tuple[object, ...]]) -> None:
            raise RuntimeError("injected insert failure")

    connection.cursor = lambda: FailingCursor(connection)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected"):
        publish_native_cache(connection, build, tables=UNIVERSAL_202411_202607_TABLES)
    connection.cursor = original  # type: ignore[method-assign]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
