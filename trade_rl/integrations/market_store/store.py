"""PostgreSQL implementation of the market-data store (requires the ``store`` extra).

This is the only module that imports ``psycopg``. Connections are autocommit; every
atomic unit (one archive, one migration) runs inside an explicit transaction.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any, Final, Self

import psycopg
from psycopg.types.json import Jsonb

from trade_rl.integrations.market_store.checksum import ChecksumEvidence
from trade_rl.integrations.market_store.contracts import (
    MARKET_STORE_URL_ENV,
    ArchiveSpec,
    EmbargoState,
    FundingRow,
    KlineRow,
    StoredExchangeInfo,
    StoreMarket,
    ms_from_utc,
    utc_from_ms,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.ingest import (
    ConflictRecord,
    IngestError,
    RunCounts,
    SourceRecord,
)
from trade_rl.integrations.market_store.rules import SymbolRule

_MIGRATION_NAME: Final = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
_MIGRATION_LOCK: Final = 810_000_001
_INGEST_LOCK: Final = 810_000_002


class MigrationError(RuntimeError):
    """The migration files and the recorded schema_version history disagree."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum_sha256: str


def migration_checksum(sql: str) -> str:
    """SHA-256 of the file text with line endings normalised to LF."""

    normalised = sql.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def load_migrations() -> tuple[Migration, ...]:
    """The shipped ``migrations/NNNN_name.sql`` files, contiguous from 0000."""

    directory = resources.files("trade_rl.integrations.market_store").joinpath(
        "migrations"
    )
    found: list[Migration] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = _MIGRATION_NAME.match(entry.name)
        if match is None:
            if entry.name.endswith(".sql"):
                raise MigrationError(f"migration file name is invalid: {entry.name}")
            continue
        sql = entry.read_text(encoding="utf-8")
        found.append(
            Migration(int(match.group(1)), match.group(2), sql, migration_checksum(sql))
        )
    found.sort(key=lambda item: item.version)
    if [item.version for item in found] != list(range(len(found))):
        raise MigrationError("migration versions must be unique and contiguous from 0")
    return tuple(found)


def resolve_url(url: str | None = None) -> str:
    resolved = url if url is not None else os.environ.get(MARKET_STORE_URL_ENV)
    if not resolved:
        raise ValueError(
            f"set {MARKET_STORE_URL_ENV} to the PostgreSQL connection URL "
            "(never pass credentials on the command line)"
        )
    return resolved


type Connection = psycopg.Connection[tuple[Any, ...]]


def connect(url: str | None = None, *, schema: str | None = None) -> Connection:
    """Open an autocommit connection (optionally pinned to one schema).

    Every timestamp crosses the wire as a timezone-aware value and is converted with
    ``astimezone(UTC)``, so the session time zone never changes a stored instant.
    """

    if schema is None:
        return psycopg.connect(resolve_url(url), autocommit=True)
    if re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema) is None:
        raise ValueError("schema must be a simple lowercase identifier")
    return psycopg.connect(
        resolve_url(url), autocommit=True, options=f"-c search_path={schema}"
    )


def migrate(connection: Connection) -> list[Migration]:
    """Apply pending migrations in order; return those applied by this call."""

    migrations = load_migrations()
    applied_now: list[Migration] = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK,))
    try:
        recorded: dict[int, str] = {}
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('schema_version') IS NOT NULL")
            exists = bool(cursor.fetchone()[0])  # type: ignore[index]
            if exists:
                cursor.execute(
                    "SELECT version, checksum_sha256 FROM schema_version ORDER BY version"
                )
                recorded = {int(row[0]): str(row[1]) for row in cursor.fetchall()}
        known = {item.version: item for item in migrations}
        unknown = sorted(set(recorded) - set(known))
        if unknown:
            raise MigrationError(
                f"database has migrations {unknown} that this code does not ship; "
                "upgrade trade_rl before using this store"
            )
        for version, checksum in recorded.items():
            if known[version].checksum_sha256 != checksum:
                raise MigrationError(
                    f"applied migration {version:04d}_{known[version].name} was "
                    "modified after it was applied; add a new migration instead"
                )
        if sorted(recorded) != list(range(len(recorded))):
            raise MigrationError("applied migration history has a gap")
        for migration in migrations:
            if migration.version in recorded:
                continue
            with connection.transaction(), connection.cursor() as cursor:
                cursor.execute(migration.sql)
                cursor.execute(
                    "INSERT INTO schema_version (version, name, checksum_sha256) "
                    "VALUES (%s, %s, %s)",
                    (migration.version, migration.name, migration.checksum_sha256),
                )
            applied_now.append(migration)
    finally:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))
    return applied_now


_KLINE_COLUMNS: Final = (
    "open_time, open, high, low, close, volume, close_time, quote_volume, trades, "
    "taker_buy_base_volume, taker_buy_quote_volume"
)


def _kline_from_row(row: Sequence[Any]) -> KlineRow:
    return KlineRow(
        open_time_ms=ms_from_utc(row[0]),
        open=row[1],
        high=row[2],
        low=row[3],
        close=row[4],
        volume=row[5],
        close_time_ms=ms_from_utc(row[6]),
        quote_volume=row[7],
        trades=int(row[8]),
        taker_buy_base_volume=row[9],
        taker_buy_quote_volume=row[10],
    )


def _funding_from_row(row: Sequence[Any]) -> FundingRow:
    return FundingRow(
        funding_time_ms=ms_from_utc(row[0]),
        rate=row[1],
        mark_price=row[2],
        interval_hours=None if row[3] is None else int(row[3]),
    )


class PostgresMarketStore:
    """The ingest store and the market reader over one PostgreSQL connection."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._holds_ingest_lock = False

    @classmethod
    def open(cls, url: str | None = None, *, schema: str | None = None) -> Self:
        return cls(connect(url, schema=schema))

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def connection(self) -> Connection:
        return self._connection

    # -- schema -----------------------------------------------------------------

    def migrate(self) -> list[Migration]:
        return migrate(self._connection)

    def schema_versions(self) -> list[tuple[int, str]]:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('schema_version') IS NOT NULL")
            row = cursor.fetchone()
            if row is None or not row[0]:
                return []
            cursor.execute("SELECT version, name FROM schema_version ORDER BY version")
            return [(int(item[0]), str(item[1])) for item in cursor.fetchall()]

    # -- ingest store -----------------------------------------------------------

    def transaction(self) -> AbstractContextManager[None]:
        @contextmanager
        def scope() -> Iterator[None]:
            with self._connection.transaction():
                yield

        return scope()

    def begin_run(
        self,
        *,
        code_commit: str,
        command_line: str,
        request: Mapping[str, object],
        embargo: EmbargoPolicy,
    ) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (_INGEST_LOCK,))
            if not bool(cursor.fetchone()[0]):  # type: ignore[index]
                raise IngestError("another ingestion run holds the store lock")
            self._holds_ingest_lock = True
            if embargo.released:
                cursor.execute(
                    "UPDATE market_embargo SET released_at = now(), "
                    "release_authorization_url = %s "
                    "WHERE id = 1 AND released_at IS NULL",
                    (embargo.authorization_url,),
                )
            cursor.execute(
                "INSERT INTO ingestion_runs (code_commit, command_line, request, "
                "embargo_released, release_authorization_url) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING run_id",
                (
                    code_commit,
                    command_line,
                    Jsonb(request),
                    embargo.released,
                    embargo.authorization_url,
                ),
            )
            return int(cursor.fetchone()[0])  # type: ignore[index]

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        counts: RunCounts,
        error: str | None,
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET finished_at = now(), status = %s, "
                "archives_planned = %s, archives_ingested = %s, archives_skipped = %s, "
                "archives_missing = %s, rows_inserted = %s, rows_identical = %s, "
                "conflicts = %s, error = %s WHERE run_id = %s",
                (
                    status,
                    counts.archives_planned,
                    counts.archives_ingested,
                    counts.archives_skipped,
                    counts.archives_missing,
                    counts.rows_inserted,
                    counts.rows_identical,
                    counts.conflicts,
                    error,
                    run_id,
                ),
            )
            if self._holds_ingest_lock:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (_INGEST_LOCK,))
                self._holds_ingest_lock = False

    def ensure_instrument(self, market: StoreMarket, symbol: str) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO instruments (exchange, market, symbol, contract_type) "
                "VALUES ('binance', %s, %s, %s) "
                "ON CONFLICT (exchange, market, symbol) DO NOTHING",
                (market.value, symbol, market.contract_type),
            )
            cursor.execute(
                "SELECT instrument_id FROM instruments "
                "WHERE exchange = 'binance' AND market = %s AND symbol = %s",
                (market.value, symbol),
            )
            return int(cursor.fetchone()[0])  # type: ignore[index]

    def source_record(self, url: str) -> SourceRecord | None:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT status, sha256 FROM sources WHERE url = %s", (url,))
            row = cursor.fetchone()
        return None if row is None else SourceRecord(str(row[0]), row[1])

    def write_source(
        self,
        *,
        run_id: int,
        spec: ArchiveSpec,
        instrument_id: int,
        http_status: int,
        fetched_at: datetime,
        evidence: ChecksumEvidence | None,
        row_count: int | None,
    ) -> int:
        ingested = evidence is not None
        with self._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sources (url, kind, instrument_id, bar_interval, "
                "period_kind, period_start, period_end, status, http_status, sha256, "
                "size_bytes, checksum_url, checksum_file_sha256, checksum_line_sha256, "
                "fetched_at, row_count, run_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s) "
                "ON CONFLICT (url) DO UPDATE SET status = EXCLUDED.status, "
                "http_status = EXCLUDED.http_status, sha256 = EXCLUDED.sha256, "
                "size_bytes = EXCLUDED.size_bytes, "
                "checksum_file_sha256 = EXCLUDED.checksum_file_sha256, "
                "checksum_line_sha256 = EXCLUDED.checksum_line_sha256, "
                "fetched_at = EXCLUDED.fetched_at, row_count = EXCLUDED.row_count, "
                "run_id = EXCLUDED.run_id RETURNING source_id",
                (
                    spec.url,
                    spec.kind.value,
                    instrument_id,
                    spec.interval,
                    spec.period_kind.value,
                    spec.period_start,
                    spec.period_end,
                    "ingested" if ingested else "missing",
                    http_status,
                    None if evidence is None else evidence.archive_sha256,
                    None if evidence is None else evidence.size_bytes,
                    spec.checksum_url,
                    None if evidence is None else evidence.checksum_file_sha256,
                    None if evidence is None else evidence.checksum_line_sha256,
                    fetched_at,
                    row_count,
                    run_id,
                ),
            )
            return int(cursor.fetchone()[0])  # type: ignore[index]

    def existing_klines(
        self, instrument_id: int, interval: str, start_ms: int, end_ms: int
    ) -> dict[int, KlineRow]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {_KLINE_COLUMNS} FROM klines WHERE instrument_id = %s "
                "AND bar_interval = %s AND open_time >= %s AND open_time < %s "
                "ORDER BY open_time",
                (instrument_id, interval, utc_from_ms(start_ms), utc_from_ms(end_ms)),
            )
            rows = [_kline_from_row(row) for row in cursor.fetchall()]
        return {row.open_time_ms: row for row in rows}

    def insert_klines(
        self,
        instrument_id: int,
        interval: str,
        source_id: int,
        rows: Sequence[KlineRow],
    ) -> int:
        if not rows:
            return 0
        with self._connection.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO klines (instrument_id, bar_interval, source_id, "
                f"{_KLINE_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s)",
                [
                    (
                        instrument_id,
                        interval,
                        source_id,
                        utc_from_ms(row.open_time_ms),
                        row.open,
                        row.high,
                        row.low,
                        row.close,
                        row.volume,
                        utc_from_ms(row.close_time_ms),
                        row.quote_volume,
                        row.trades,
                        row.taker_buy_base_volume,
                        row.taker_buy_quote_volume,
                    )
                    for row in rows
                ],
            )
        return len(rows)

    def existing_funding(
        self, instrument_id: int, start_ms: int, end_ms: int
    ) -> dict[int, FundingRow]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT funding_time, rate, mark_price, funding_interval_hours "
                "FROM funding_rates WHERE instrument_id = %s AND funding_time >= %s "
                "AND funding_time < %s ORDER BY funding_time",
                (instrument_id, utc_from_ms(start_ms), utc_from_ms(end_ms)),
            )
            rows = [_funding_from_row(row) for row in cursor.fetchall()]
        return {row.funding_time_ms: row for row in rows}

    def insert_funding(
        self, instrument_id: int, source_id: int, rows: Sequence[FundingRow]
    ) -> int:
        if not rows:
            return 0
        with self._connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO funding_rates (instrument_id, source_id, funding_time, "
                "rate, mark_price, funding_interval_hours) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        instrument_id,
                        source_id,
                        utc_from_ms(row.funding_time_ms),
                        row.rate,
                        row.mark_price,
                        row.interval_hours,
                    )
                    for row in rows
                ],
            )
        return len(rows)

    def record_conflicts(
        self, run_id: int, conflicts: Sequence[ConflictRecord]
    ) -> None:
        for record in conflicts:
            instrument_id = self.ensure_instrument(record.market, record.symbol)
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO ingestion_conflicts (run_id, kind, instrument_id, "
                    "source_url, bar_interval, key_time, column_name, existing_value, "
                    "incoming_value) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        run_id,
                        record.kind,
                        instrument_id,
                        record.source_url,
                        record.bar_interval,
                        None
                        if record.key_time_ms is None
                        else utc_from_ms(record.key_time_ms),
                        record.column,
                        record.existing,
                        record.incoming,
                    ),
                )

    # -- exchange rules -----------------------------------------------------------

    def store_exchange_info(
        self,
        info: StoredExchangeInfo,
        rules: Sequence[SymbolRule],
        *,
        run_id: int | None,
    ) -> bool:
        """Store a raw snapshot and its per-symbol rules; False if already stored."""

        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO exchange_info_snapshots (exchange, market, retrieved_at, "
                "source_uri, raw_payload_sha256, raw_payload, run_id) "
                "VALUES ('binance', %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (exchange, market, raw_payload_sha256) DO NOTHING "
                "RETURNING snapshot_id",
                (
                    info.market.value,
                    info.retrieved_at,
                    info.source_uri,
                    info.raw_payload_sha256,
                    info.raw_payload,
                    run_id,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is None:
                return False
            snapshot_id = int(inserted[0])
            for rule in rules:
                instrument_id = self.ensure_instrument(info.market, rule.symbol)
                cursor.execute(
                    "INSERT INTO instrument_rule_snapshots (instrument_id, snapshot_id, "
                    "valid_from, trading_status, onboard_date, tick_size, lot_size, "
                    "min_notional) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        instrument_id,
                        snapshot_id,
                        info.retrieved_at,
                        rule.trading_status,
                        rule.onboard_date,
                        rule.tick_size,
                        rule.lot_size,
                        rule.min_notional,
                    ),
                )
        return True

    # -- reader -----------------------------------------------------------------

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {', '.join('k.' + c.strip() for c in _KLINE_COLUMNS.split(','))} "
                "FROM klines k JOIN instruments i USING (instrument_id) "
                "WHERE i.exchange = 'binance' AND i.market = %s AND i.symbol = %s "
                "AND k.bar_interval = %s AND k.open_time >= %s AND k.open_time < %s "
                "ORDER BY k.open_time",
                (
                    market.value,
                    symbol,
                    interval,
                    utc_from_ms(start_ms),
                    utc_from_ms(end_ms),
                ),
            )
            return [_kline_from_row(row) for row in cursor.fetchall()]

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT f.funding_time, f.rate, f.mark_price, f.funding_interval_hours "
                "FROM funding_rates f JOIN instruments i USING (instrument_id) "
                "WHERE i.exchange = 'binance' AND i.market = %s AND i.symbol = %s "
                "AND f.funding_time >= %s AND f.funding_time <= %s "
                "ORDER BY f.funding_time",
                (market.value, symbol, utc_from_ms(start_ms), utc_from_ms(end_ms)),
            )
            return [_funding_from_row(row) for row in cursor.fetchall()]

    def load_exchange_info(
        self, market: StoreMarket, sha256: str | None = None
    ) -> StoredExchangeInfo:
        with self._connection.cursor() as cursor:
            if sha256 is None:
                cursor.execute(
                    "SELECT raw_payload, raw_payload_sha256, source_uri, retrieved_at "
                    "FROM exchange_info_snapshots WHERE exchange = 'binance' "
                    "AND market = %s ORDER BY retrieved_at DESC, snapshot_id DESC "
                    "LIMIT 1",
                    (market.value,),
                )
            else:
                cursor.execute(
                    "SELECT raw_payload, raw_payload_sha256, source_uri, retrieved_at "
                    "FROM exchange_info_snapshots WHERE exchange = 'binance' "
                    "AND market = %s AND raw_payload_sha256 = %s",
                    (market.value, sha256),
                )
            row = cursor.fetchone()
        if row is None:
            raise LookupError(
                f"no stored exchangeInfo snapshot for {market.value}"
                + ("" if sha256 is None else f" with sha256 {sha256}")
            )
        raw = bytes(row[0])
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise RuntimeError("stored exchangeInfo payload does not match its digest")
        return StoredExchangeInfo(
            market=market,
            raw_payload=raw,
            raw_payload_sha256=str(row[1]),
            source_uri=str(row[2]),
            retrieved_at=row[3],
        )

    def embargo_state(self) -> EmbargoState:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT sealed_from, released_at, release_authorization_url "
                "FROM market_embargo WHERE id = 1"
            )
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("market_embargo row is missing; run migrate")
        return EmbargoState(
            sealed_from=row[0],
            released=row[1] is not None,
            authorization_url=row[2],
        )

    # -- status -----------------------------------------------------------------

    def status(self) -> dict[str, object]:
        versions = self.schema_versions()
        if not versions:
            return {"schema_versions": [], "migrated": False}
        state = self.embargo_state()
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT i.market, i.symbol, k.bar_interval, count(*), "
                "min(k.open_time), max(k.open_time) FROM klines k "
                "JOIN instruments i USING (instrument_id) "
                "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
            )
            klines = [
                {
                    "market": str(r[0]),
                    "symbol": str(r[1]),
                    "interval": str(r[2]),
                    "rows": int(r[3]),
                    "first_open_time": r[4].isoformat(),
                    "last_open_time": r[5].isoformat(),
                }
                for r in cursor.fetchall()
            ]
            cursor.execute(
                "SELECT i.market, i.symbol, count(*), min(f.funding_time), "
                "max(f.funding_time) FROM funding_rates f "
                "JOIN instruments i USING (instrument_id) GROUP BY 1, 2 ORDER BY 1, 2"
            )
            funding = [
                {
                    "market": str(r[0]),
                    "symbol": str(r[1]),
                    "rows": int(r[2]),
                    "first_funding_time": r[3].isoformat(),
                    "last_funding_time": r[4].isoformat(),
                }
                for r in cursor.fetchall()
            ]
            cursor.execute("SELECT status, count(*) FROM sources GROUP BY 1 ORDER BY 1")
            sources = {str(r[0]): int(r[1]) for r in cursor.fetchall()}
            cursor.execute("SELECT count(*) FROM ingestion_conflicts")
            conflicts = int(cursor.fetchone()[0])  # type: ignore[index]
            cursor.execute(
                "SELECT run_id, status, started_at, archives_ingested, rows_inserted, "
                "conflicts FROM ingestion_runs ORDER BY run_id DESC LIMIT 5"
            )
            runs = [
                {
                    "run_id": int(r[0]),
                    "status": str(r[1]),
                    "started_at": r[2].isoformat(),
                    "archives_ingested": int(r[3]),
                    "rows_inserted": int(r[4]),
                    "conflicts": int(r[5]),
                }
                for r in cursor.fetchall()
            ]
        return {
            "migrated": True,
            "schema_versions": [list(item) for item in versions],
            "embargo": {
                "sealed_from": state.sealed_from.isoformat(),
                "released": state.released,
                "authorization_url": state.authorization_url,
            },
            "klines": klines,
            "funding_rates": funding,
            "sources": sources,
            "recorded_conflicts": conflicts,
            "latest_runs": runs,
        }


__all__ = [
    "Connection",
    "Migration",
    "MigrationError",
    "PostgresMarketStore",
    "connect",
    "load_migrations",
    "migrate",
    "migration_checksum",
    "resolve_url",
]
