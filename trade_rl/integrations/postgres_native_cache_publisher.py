"""Transactional publication for immutable native PostgreSQL cache generations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.integrations.postgres_market_tables import PostgresMarketTableSet
from trade_rl.workflows.native_indicator_materializer import NativeCacheBuild


class _Cursor(Protocol):
    def execute(self, query: str, params: object = None) -> Any: ...

    def executemany(self, query: str, params: list[tuple[object, ...]]) -> Any: ...

    def fetchone(self) -> Sequence[object] | None: ...

    def fetchall(self) -> Sequence[Sequence[object]]: ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *args: object) -> None: ...


class NativeCacheConnection(Protocol):
    """DB-API transaction boundary used by cache publication."""

    def cursor(self) -> _Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishedNativeCache:
    """Verified database identity and row counts for one publication."""

    cache_id: str
    manifest_digest: str
    artifact_count: int
    kline_row_count: int
    funding_row_count: int
    tables: PostgresMarketTableSet

    @classmethod
    def from_build(
        cls, build: NativeCacheBuild, *, tables: PostgresMarketTableSet
    ) -> PublishedNativeCache:
        return cls(
            cache_id=build.manifest.cache_id,
            manifest_digest=build.manifest.digest,
            artifact_count=len(build.artifacts),
            kline_row_count=sum(
                len(bars.open_time_ms) for bars in build.market_bars.values()
            ),
            funding_row_count=sum(
                int(bars.funding_available.sum())
                for (_, timeframe), bars in build.market_bars.items()
                if timeframe == "15m"
            ),
            tables=tables,
        )


def _create_tables(cursor: _Cursor, *, tables: PostgresMarketTableSet) -> None:
    cursor.execute("CREATE SCHEMA IF NOT EXISTS market_raw")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.kline} (
            cache_id text NOT NULL,
            symbol text NOT NULL,
            interval text NOT NULL,
            open_time_ms bigint NOT NULL,
            event_time_ms bigint NOT NULL,
            open double precision NOT NULL,
            high double precision NOT NULL,
            low double precision NOT NULL,
            close double precision NOT NULL,
            quote_volume double precision NOT NULL,
            PRIMARY KEY (cache_id, symbol, interval, open_time_ms)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.funding} (
            cache_id text NOT NULL,
            symbol text NOT NULL,
            calc_time_ms bigint NOT NULL,
            last_funding_rate double precision NOT NULL,
            PRIMARY KEY (cache_id, symbol, calc_time_ms)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.indicator_manifest} (
            cache_id text PRIMARY KEY,
            schema_version text NOT NULL,
            market text NOT NULL,
            symbols text[] NOT NULL,
            start_time timestamptz NOT NULL,
            end_time timestamptz NOT NULL,
            feature_config_digest char(64) NOT NULL,
            feature_specs jsonb NOT NULL,
            artifact_count integer NOT NULL,
            volume_conversion_method text NOT NULL,
            manifest_digest char(64) NOT NULL
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tables.indicator_artifact} (
            cache_id text NOT NULL,
            symbol text NOT NULL,
            timeframe text NOT NULL,
            row_count integer NOT NULL,
            feature_count integer NOT NULL,
            available_value_count bigint NOT NULL,
            first_event_time_ms bigint,
            last_event_time_ms bigint,
            payload_schema text NOT NULL,
            payload_sha256 char(64) NOT NULL,
            payload_bytes bigint NOT NULL,
            npz_payload bytea NOT NULL,
            PRIMARY KEY (cache_id, symbol, timeframe)
        )
        """
    )


def _batches(
    rows: Iterable[tuple[object, ...]], *, size: int = 5_000
) -> Iterable[list[tuple[object, ...]]]:
    batch: list[tuple[object, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _kline_rows(build: NativeCacheBuild) -> Iterable[tuple[object, ...]]:
    for (symbol, timeframe), bars in build.market_bars.items():
        for index in range(len(bars.open_time_ms)):
            yield (
                build.manifest.cache_id,
                symbol,
                timeframe,
                int(bars.open_time_ms[index]),
                int(bars.event_time_ms[index]),
                float(bars.open[index]),
                float(bars.high[index]),
                float(bars.low[index]),
                float(bars.close[index]),
                float(bars.quote_volume[index]),
            )


def _funding_rows(build: NativeCacheBuild) -> Iterable[tuple[object, ...]]:
    for (symbol, timeframe), bars in build.market_bars.items():
        if timeframe != "15m":
            continue
        for index in bars.funding_available.nonzero()[0]:
            yield (
                build.manifest.cache_id,
                symbol,
                int(bars.event_time_ms[index]),
                float(bars.funding_rate[index]),
            )


def _insert_market_rows(
    cursor: _Cursor, *, build: NativeCacheBuild, tables: PostgresMarketTableSet
) -> None:
    kline_sql = f"""
        INSERT INTO {tables.kline} (
            cache_id, symbol, interval, open_time_ms, event_time_ms,
            open, high, low, close, quote_volume
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for batch in _batches(_kline_rows(build)):
        cursor.executemany(kline_sql, batch)
    funding_sql = f"""
        INSERT INTO {tables.funding} (
            cache_id, symbol, calc_time_ms, last_funding_rate
        ) VALUES (%s, %s, %s, %s)
    """
    for batch in _batches(_funding_rows(build)):
        cursor.executemany(funding_sql, batch)


def _insert_indicator_rows(
    cursor: _Cursor, *, build: NativeCacheBuild, tables: PostgresMarketTableSet
) -> None:
    cursor.execute(
        f"""
        INSERT INTO {tables.indicator_manifest} (
            cache_id, schema_version, market, symbols, start_time, end_time,
            feature_config_digest, feature_specs, artifact_count,
            volume_conversion_method, manifest_digest
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
        """,
        (
            build.manifest.cache_id,
            "native_indicator_cache_v1",
            "usds-m",
            list(build.manifest.symbols),
            build.manifest.start_time,
            build.manifest.end_time,
            build.manifest.feature_config_digest,
            canonical_json_bytes(dict(build.manifest.feature_specs)).decode("utf-8"),
            len(build.artifacts),
            build.manifest.volume_conversion_method,
            build.manifest.digest,
        ),
    )
    rows: list[tuple[object, ...]] = [
        (
            build.manifest.cache_id,
            artifact.symbol,
            artifact.timeframe,
            artifact.row_count,
            artifact.feature_count,
            artifact.available_value_count,
            None if not artifact.row_count else int(artifact.event_time_ms[0]),
            None if not artifact.row_count else int(artifact.event_time_ms[-1]),
            artifact.payload_schema,
            artifact.payload_sha256,
            len(artifact.payload),
            artifact.payload,
        )
        for artifact in build.artifacts
    ]
    cursor.executemany(
        f"""
        INSERT INTO {tables.indicator_artifact} (
            cache_id, symbol, timeframe, row_count, feature_count,
            available_value_count, first_event_time_ms, last_event_time_ms,
            payload_schema, payload_sha256, payload_bytes, npz_payload
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )


def _verify_published_cache(
    cursor: _Cursor, *, build: NativeCacheBuild, tables: PostgresMarketTableSet
) -> None:
    expected = PublishedNativeCache.from_build(build, tables=tables)
    cursor.execute(
        f"""
        SELECT manifest_digest, artifact_count
        FROM {tables.indicator_manifest}
        WHERE cache_id = %s
        """,
        (expected.cache_id,),
    )
    manifest = cursor.fetchone()
    if manifest is None or tuple(manifest) != (
        expected.manifest_digest,
        expected.artifact_count,
    ):
        raise FileExistsError("cache identity already exists with different content")
    cursor.execute(
        f"""
        SELECT symbol, timeframe, payload_sha256
        FROM {tables.indicator_artifact}
        WHERE cache_id = %s
        ORDER BY symbol, timeframe
        """,
        (expected.cache_id,),
    )
    observed_artifacts = tuple(tuple(row) for row in cursor.fetchall())
    expected_artifacts = tuple(
        sorted(
            (artifact.symbol, artifact.timeframe, artifact.payload_sha256)
            for artifact in build.artifacts
        )
    )
    if observed_artifacts != expected_artifacts:
        raise FileExistsError("cache identity already exists with different content")
    for table, expected_count in (
        (tables.kline, expected.kline_row_count),
        (tables.funding, expected.funding_row_count),
    ):
        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE cache_id = %s",
            (expected.cache_id,),
        )
        count_row = cursor.fetchone()
        if count_row is None or tuple(count_row) != (expected_count,):
            raise FileExistsError(
                "cache identity already exists with different content"
            )


def publish_native_cache(
    connection: NativeCacheConnection,
    build: NativeCacheBuild,
    *,
    tables: PostgresMarketTableSet,
) -> PublishedNativeCache:
    """Atomically publish or fully verify one immutable cache identity."""

    try:
        with connection.cursor() as cursor:
            _create_tables(cursor, tables=tables)
            cursor.execute(
                f"""
                SELECT manifest_digest
                FROM {tables.indicator_manifest}
                WHERE cache_id = %s
                FOR UPDATE
                """,
                (build.manifest.cache_id,),
            )
            existing = cursor.fetchone()
            if existing is None:
                _insert_market_rows(cursor, build=build, tables=tables)
                _insert_indicator_rows(cursor, build=build, tables=tables)
            elif tuple(existing) != (build.manifest.digest,):
                raise FileExistsError(
                    "cache identity already exists with different content"
                )
            _verify_published_cache(cursor, build=build, tables=tables)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return PublishedNativeCache.from_build(build, tables=tables)


__all__ = [
    "NativeCacheConnection",
    "PublishedNativeCache",
    "publish_native_cache",
]
