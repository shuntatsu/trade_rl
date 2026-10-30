"""Real-PostgreSQL contract tests (skipped unless TRADE_RL_TEST_POSTGRES_URL is set)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.integrations.market_store.support import (
    HOUR_MS,
    NOW,
    SYMBOLS,
    FakeFetcher,
    archive_bytes,
    fetcher_for,
    file_backed_build,
    kline_csv_rows,
    stored_exchange_info,
    zip_csv,
)
from trade_rl.integrations.market_store.contracts import (
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    SEALED_WINDOW_START,
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.export import export_market_dataset
from trade_rl.integrations.market_store.ingest import (
    ConflictError,
    IngestReport,
    RunCounts,
    run_ingestion,
)
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion
from trade_rl.integrations.market_store.rules import parse_symbol_rules

URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-31"
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 2, 1, tzinfo=UTC)


def plan_for(markets: tuple[StoreMarket, ...]) -> IngestionPlan:
    return plan_ingestion(
        Universe(symbols=SYMBOLS, markets=markets), start=START, end=END, now=NOW
    )


def ingest(
    store: Any, fetcher: FakeFetcher, plan: IngestionPlan, **kwargs: Any
) -> IngestReport:
    return run_ingestion(
        store,
        fetcher,
        plan,
        code_commit="deadbeef",
        command_line="pytest",
        clock=lambda: NOW,
        **kwargs,
    )


def scalar(store: Any, sql: str, *params: Any) -> Any:
    with store.connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchone()[0]


# -- migrations -----------------------------------------------------------------


def test_migrate_applies_in_order_and_is_idempotent(
    postgres_url: str, pg_schema: str
) -> None:
    from trade_rl.integrations.market_store.store import (
        PostgresMarketStore,
        load_migrations,
    )

    shipped = load_migrations()
    assert [m.version for m in shipped] == list(range(len(shipped)))
    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as store:
        assert store.schema_versions() == []
        applied = store.migrate()
        assert [m.version for m in applied] == list(range(len(shipped)))
        assert store.migrate() == []
        assert [v for v, _ in store.schema_versions()] == list(range(len(shipped)))
        assert scalar(store, "SELECT count(*) FROM schema_version") == len(shipped)
        stored = scalar(
            store, "SELECT checksum_sha256 FROM schema_version WHERE version = 1"
        )
        assert stored == shipped[1].checksum_sha256


def test_migrate_refuses_a_modified_applied_migration_and_a_newer_database(
    pg_store: Any,
) -> None:
    from trade_rl.integrations.market_store.store import MigrationError

    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "UPDATE schema_version SET checksum_sha256 = %s WHERE version = 1",
            ("0" * 64,),
        )
    with pytest.raises(MigrationError, match="modified"):
        pg_store.migrate()
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "UPDATE schema_version SET checksum_sha256 = "
            "(SELECT checksum_sha256 FROM schema_version WHERE version = 0) WHERE version = 1"
        )
    # restore the true checksum of migration 1 by recomputing it from the shipped file
    from trade_rl.integrations.market_store.store import load_migrations

    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "UPDATE schema_version SET checksum_sha256 = %s WHERE version = 1",
            (load_migrations()[1].checksum_sha256,),
        )
        cursor.execute(
            "INSERT INTO schema_version (version, name, checksum_sha256) VALUES (99, 'future', %s)",
            ("1" * 64,),
        )
    with pytest.raises(MigrationError, match="does not ship"):
        pg_store.migrate()


# -- constraints ----------------------------------------------------------------


def seed_instrument_and_source(store: Any) -> tuple[int, int]:
    import psycopg

    instrument_id = store.ensure_instrument(StoreMarket.SPOT, "BTCUSDT")
    with store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO ingestion_runs (code_commit, command_line, request) "
            "VALUES ('c', 'c', '{}'::jsonb) RETURNING run_id"
        )
        run_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
            "period_start, period_end, status, http_status, sha256, size_bytes, "
            "checksum_url, checksum_file_sha256, checksum_line_sha256, fetched_at, "
            "row_count, run_id) VALUES ('u', 'klines', %s, '1h', 'monthly', "
            "'2024-01-01+00', '2024-02-01+00', 'ingested', 200, %s, 1, 'u.CHECKSUM', "
            "%s, %s, now(), 1, %s) RETURNING source_id",
            (instrument_id, "a" * 64, "b" * 64, "a" * 64, run_id),
        )
        source_id = cursor.fetchone()[0]
    assert psycopg is not None
    return instrument_id, source_id


KLINE_SQL = (
    "INSERT INTO klines (instrument_id, bar_interval, open_time, open, high, low, "
    "close, volume, close_time, quote_volume, trades, taker_buy_base_volume, "
    "taker_buy_quote_volume, source_id) VALUES (%s, '1h', %s, %s, %s, %s, %s, %s, "
    "%s, 1, 1, 0, 0, %s)"
)


def insert_kline(
    store: Any,
    instrument_id: int,
    source_id: int,
    *,
    open_time: str = "2024-01-01 00:00+00",
    close_time: str = "2024-01-01 00:59:59.999+00",
    prices: tuple[str, str, str, str] = ("10", "12", "9", "11"),
    volume: str = "1",
) -> None:
    with store.connection.cursor() as cursor:
        cursor.execute(
            KLINE_SQL,
            (
                instrument_id,
                open_time,
                *prices,
                volume,
                close_time,
                source_id,
            ),
        )


def test_constraints_reject_invalid_klines(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    insert_kline(pg_store, instrument_id, source_id)
    bad_cases = {
        "high below close": ("10", "10.5", "9", "11"),
        "high below open": ("12", "11", "9", "10"),
        "low above open": ("10", "12", "10.5", "11"),
        "low above close": ("10", "12", "11.5", "11"),
        "non-positive price": ("0", "12", "0", "11"),
    }
    for index, (label, prices) in enumerate(bad_cases.items(), start=1):
        with pytest.raises(psycopg.errors.CheckViolation):
            (
                insert_kline(
                    pg_store,
                    instrument_id,
                    source_id,
                    open_time=f"2024-01-01 {index:02d}:00+00",
                    close_time=f"2024-01-01 {index:02d}:59:59.999+00",
                    prices=prices,
                ),
                label,
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time="2024-01-01 09:00+00",
            close_time="2024-01-01 09:59:59.999+00",
            volume="-1",
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time="2024-01-01 10:00+00",
            close_time="2024-01-01 10:00+00",
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_kline(pg_store, instrument_id, source_id)


def test_rows_are_append_only_and_instrument_rules_hold(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    insert_kline(pg_store, instrument_id, source_id)
    with pytest.raises(psycopg.errors.RestrictViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute("UPDATE klines SET close = 11.5")
    with pytest.raises(psycopg.errors.RestrictViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute("DELETE FROM klines")
    with pytest.raises(psycopg.errors.RestrictViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute("UPDATE sources SET row_count = 5")
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO instruments (exchange, market, symbol, contract_type) "
                "VALUES ('binance', 'usds_m', 'XYZUSDT', NULL)"
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO instruments (exchange, market, symbol, contract_type) "
                "VALUES ('binance', 'spot', 'XYZUSDT', 'PERPETUAL')"
            )


def test_numeric_values_round_trip_exactly(pg_store: Any) -> None:
    instrument_id, source_id = seed_instrument_and_source(pg_store)
    exact = ("12345678.12345678", "99999999.99999999", "0.00000001", "0.1")
    insert_kline(pg_store, instrument_id, source_id, prices=exact, volume="0.30000000")
    rows = pg_store.read_klines(StoreMarket.SPOT, "BTCUSDT", "1h", 0, 4_000_000_000_000)
    assert len(rows) == 1
    row = rows[0]
    assert (row.open, row.high, row.low, row.close) == tuple(Decimal(v) for v in exact)
    assert format(row.open, "f") == "12345678.12345678"
    assert row.volume == Decimal("0.3") and format(row.volume, "f") == "0.30000000"
    assert scalar(pg_store, "SELECT 0.1::numeric + 0.2::numeric = 0.3::numeric") is True


# -- ingestion ------------------------------------------------------------------


def test_ingest_is_idempotent_and_records_evidence(pg_store: Any) -> None:
    plan = plan_for((StoreMarket.USDS_M, StoreMarket.SPOT))
    fetcher = fetcher_for(plan)
    first = ingest(pg_store, fetcher, plan)
    assert first.counts.archives_ingested == len(plan.specs) == 6
    klines = scalar(pg_store, "SELECT count(*) FROM klines")
    funding = scalar(pg_store, "SELECT count(*) FROM funding_rates")
    assert klines == 31 * 24 * 4 and funding == 93 * 2
    assert first.counts.rows_inserted == klines + funding

    fetcher.requested.clear()
    second = ingest(pg_store, fetcher, plan)
    assert fetcher.requested == []
    assert second.counts.archives_skipped == 6
    assert second.counts.rows_inserted == 0
    assert scalar(pg_store, "SELECT count(*) FROM klines") == klines

    refetched = ingest(pg_store, fetcher, plan, refetch=True)
    assert (
        refetched.counts.archives_skipped == 6 and refetched.counts.rows_inserted == 0
    )
    assert scalar(pg_store, "SELECT count(*) FROM klines") == klines

    spec = plan.specs[0]
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT status, http_status, sha256, checksum_line_sha256, "
            "checksum_file_sha256, size_bytes, row_count, kind FROM sources WHERE url = %s",
            (spec.url,),
        )
        status, http, sha, line, file_sha, size, count, kind = cursor.fetchone()
    payload = fetcher.files[spec.url]
    assert (status, http, kind) == ("ingested", 200, "klines")
    assert sha == line == hashlib.sha256(payload).hexdigest()
    assert file_sha == hashlib.sha256(fetcher.files[f"{spec.url}.CHECKSUM"]).hexdigest()
    assert size == len(payload) and count == 31 * 24
    assert (
        scalar(
            pg_store, "SELECT count(*) FROM ingestion_runs WHERE status = 'succeeded'"
        )
        == 3
    )
    assert (
        scalar(
            pg_store,
            "SELECT code_commit FROM ingestion_runs WHERE run_id = %s",
            first.run_id,
        )
        == "deadbeef"
    )


def test_changed_value_is_refused_recorded_and_never_overwritten(pg_store: Any) -> None:
    plan = plan_for((StoreMarket.SPOT,))
    ingest(pg_store, fetcher_for(plan), plan)
    before = scalar(
        pg_store,
        "SELECT sum(close) FROM klines",
    )
    spec = plan.specs[0]

    changed = FakeFetcher()
    changed.add(
        spec.url,
        zip_csv(
            kline_csv_rows(
                spec.symbol,
                ms_from_utc(spec.period_start),
                ms_from_utc(spec.period_end),
                close_bump=1,
            )
        ),
    )
    with pytest.raises(ConflictError) as archive_conflict:
        ingest(pg_store, changed, plan, refetch=True)
    assert archive_conflict.value.conflicts[0].kind == "archive_sha256"

    # same keys through a different (daily) source with a changed close value
    from trade_rl.integrations.market_store.contracts import ArchiveSpec, PeriodKind

    daily_url = spec.url.replace("monthly", "daily").replace(
        "2024-01.zip", "2024-01-01.zip"
    )
    daily = ArchiveSpec(
        kind=spec.kind,
        market=spec.market,
        symbol=spec.symbol,
        interval=spec.interval,
        period_kind=PeriodKind.DAILY,
        period_start=START,
        period_end=datetime(2024, 1, 2, tzinfo=UTC),
        url=daily_url,
        checksum_url=f"{daily_url}.CHECKSUM",
    )
    fetcher = FakeFetcher()
    fetcher.add(
        daily.url,
        zip_csv(
            kline_csv_rows(
                daily.symbol,
                ms_from_utc(daily.period_start),
                ms_from_utc(daily.period_end),
                close_bump=1,
            )
        ),
    )
    daily_plan = IngestionPlan(
        plan.universe, START, daily.period_end, EmbargoPolicy.enforced(), (daily,), ()
    )
    with pytest.raises(ConflictError, match="nothing was overwritten"):
        ingest(pg_store, fetcher, daily_plan)

    assert scalar(pg_store, "SELECT sum(close) FROM klines") == before
    assert (
        scalar(pg_store, "SELECT count(*) FROM sources WHERE url = %s", daily.url) == 0
    )
    assert scalar(pg_store, "SELECT count(*) FROM ingestion_conflicts") >= 2
    assert (
        scalar(
            pg_store,
            "SELECT count(*) FROM ingestion_conflicts WHERE kind = 'kline' "
            "AND column_name = 'close' AND existing_value <> incoming_value",
        )
        == 24
    )
    assert (
        scalar(pg_store, "SELECT count(*) FROM ingestion_runs WHERE status = 'failed'")
        == 2
    )
    assert scalar(pg_store, "SELECT sum(conflicts) FROM ingestion_runs") >= 25


def test_a_404_source_is_replaced_when_the_archive_is_published(pg_store: Any) -> None:
    plan = plan_for((StoreMarket.SPOT,))
    fetcher = fetcher_for(plan)
    gone = plan.specs[0]
    published = fetcher.files.pop(gone.url)
    report = ingest(pg_store, fetcher, plan)
    assert report.counts.archives_missing == 1
    assert (
        scalar(pg_store, "SELECT status FROM sources WHERE url = %s", gone.url)
        == "missing"
    )
    fetcher.files[gone.url] = published
    again = ingest(pg_store, fetcher, plan)
    assert again.counts.archives_ingested == 1
    assert (
        scalar(pg_store, "SELECT status FROM sources WHERE url = %s", gone.url)
        == "ingested"
    )


def test_a_second_concurrent_run_is_refused(
    postgres_url: str, pg_schema: str, pg_store: Any
) -> None:
    from trade_rl.integrations.market_store.ingest import IngestError
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    first = pg_store.begin_run(
        code_commit="c", command_line="c", request={}, embargo=EmbargoPolicy.enforced()
    )
    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as other:
        with pytest.raises(IngestError, match="another ingestion"):
            other.begin_run(
                code_commit="c",
                command_line="c",
                request={},
                embargo=EmbargoPolicy.enforced(),
            )
    from trade_rl.integrations.market_store.ingest import RunCounts

    pg_store.finish_run(first, status="succeeded", counts=RunCounts(), error=None)


# -- FS1 boundaries at the database layer -----------------------------------------------

SEALED = "2026-11-01 00:00:00+00"
WARMUP = "2026-08-01 00:00:00+00"
JULY_LAST = ("2026-07-31 23:00+00", "2026-07-31 23:59:59.999+00")


def insert_funding(store: Any, instrument_id: int, source_id: int, when: str) -> None:
    with store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO funding_rates (instrument_id, funding_time, rate, source_id) "
            "VALUES (%s, %s, 0.0001, %s)",
            (instrument_id, when, source_id),
        )


def insert_source(
    store: Any, instrument_id: int, name: str, start: str, end: str
) -> None:
    with store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
            "period_start, period_end, status, http_status, checksum_url, fetched_at, "
            "run_id) SELECT %s, 'klines', %s, '1h', 'monthly', %s, %s, 'missing', 404, "
            "%s, now(), run_id FROM ingestion_runs LIMIT 1",
            (name, instrument_id, start, end, f"{name}.CHECKSUM"),
        )


def test_database_rejects_sealed_rows_even_if_python_is_bypassed(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time=SEALED,
            close_time="2026-11-01 00:59:59.999+00",
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time="2026-10-31 23:00+00",
            close_time="2026-11-01 00:00+00",
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_funding(pg_store, instrument_id, source_id, SEALED)
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_source(pg_store, instrument_id, "m", "2026-11-01+00", "2026-12-01+00")
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_source(pg_store, instrument_id, "wide", "2026-07-01+00", "2026-11-02+00")
    assert scalar(pg_store, "SELECT count(*) FROM klines") == 0


def test_database_rejects_warmup_rows_without_an_authorized_run(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    insert_kline(
        pg_store,
        instrument_id,
        source_id,
        open_time=JULY_LAST[0],
        close_time=JULY_LAST[1],
    )
    insert_funding(pg_store, instrument_id, source_id, "2026-07-31 16:00:00.005+00")
    insert_source(pg_store, instrument_id, "july", "2026-07-01+00", "2026-08-01+00")
    for when in (WARMUP, "2026-09-15 00:00+00", "2026-10-31 23:00+00"):
        with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
            insert_kline(
                pg_store,
                instrument_id,
                source_id,
                open_time=when,
                close_time=f"{when[:10]} 23:59:59.999+00",
            )
        with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
            insert_funding(pg_store, instrument_id, source_id, when)
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
        insert_source(pg_store, instrument_id, "aug", "2026-08-01+00", "2026-09-01+00")

    # a RUNNING run that was NOT authorized does not open the span either, even if the
    # session variable names it
    run_id = pg_store.begin_run(
        code_commit="c", command_line="c", request={}, embargo=EmbargoPolicy.enforced()
    )
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT set_config('trade_rl.fs1_warmup_run_id', %s, false)", (str(run_id),)
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
        insert_funding(pg_store, instrument_id, source_id, WARMUP)
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)


def test_authorized_run_admits_only_the_warmup_span_and_only_while_running(
    pg_store: Any,
) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    run_id = pg_store.begin_run(
        code_commit="c",
        command_line="c --allow-fs1-warmup-span",
        request={},
        embargo=EmbargoPolicy.with_warmup(URL),
    )
    insert_funding(pg_store, instrument_id, source_id, WARMUP)
    insert_kline(
        pg_store,
        instrument_id,
        source_id,
        open_time="2026-10-31 23:00+00",
        close_time="2026-10-31 23:59:59.999+00",
    )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_funding(pg_store, instrument_id, source_id, SEALED)
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time=SEALED,
            close_time="2026-11-01 00:59:59.999+00",
        )
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
        insert_funding(pg_store, instrument_id, source_id, "2026-09-01 00:00:00+00")
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT fs1_warmup_allowed, fs1_warmup_authorization_url "
            "FROM ingestion_runs WHERE run_id = %s",
            (run_id,),
        )
        assert cursor.fetchone() == (True, URL)


def test_ingestion_runs_record_the_warmup_flag_and_url_consistently(
    pg_store: Any,
) -> None:
    import psycopg

    insert = (
        "INSERT INTO ingestion_runs (code_commit, command_line, request, "
        "fs1_warmup_allowed, fs1_warmup_authorization_url) "
        "VALUES ('c', 'c', '{}'::jsonb, %s, %s)"
    )
    for allowed, url in (
        (True, None),
        (False, URL),
        (True, "https://example.com/not-the-issue"),
        (True, "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1"),
    ):
        with pytest.raises(psycopg.errors.CheckViolation):
            with pg_store.connection.cursor() as cursor:
                cursor.execute(insert, (allowed, url))
    with pg_store.connection.cursor() as cursor:
        cursor.execute(insert, (True, URL))
        cursor.execute(insert, (False, None))


def test_there_is_no_release_path_in_the_database(pg_store: Any) -> None:
    import psycopg

    assert scalar(pg_store, "SELECT to_regclass('market_embargo') IS NULL") is True
    assert scalar(pg_store, "SELECT market_sealed_from()") == SEALED_WINDOW_START
    assert scalar(pg_store, "SELECT market_fs1_warmup_start()") == FS1_WARMUP_START
    names = [
        row[0]
        for row in _rows(
            pg_store,
            "SELECT proname FROM pg_proc WHERE pronamespace = "
            "(SELECT oid FROM pg_namespace WHERE nspname = current_schema())",
        )
    ]
    assert not [name for name in names if "release" in name]
    state = pg_store.embargo_state()
    assert state.sealed_from == SEALED_WINDOW_START
    assert state.fs1_warmup_start == FS1_WARMUP_START
    assert not hasattr(state, "released")

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    # an authorized warm-up run, and even a forged session variable, never open the
    # sealed window
    run_id = pg_store.begin_run(
        code_commit="c",
        command_line="c",
        request={},
        embargo=EmbargoPolicy.with_warmup(URL),
    )
    for forged in (str(run_id), "0", "1", "-1", "x"):
        try:
            with pg_store.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config('trade_rl.fs1_warmup_run_id', %s, false)",
                    (forged,),
                )
            insert_funding(pg_store, instrument_id, source_id, SEALED)
        except (
            psycopg.errors.CheckViolation,
            psycopg.errors.InvalidTextRepresentation,
        ):
            continue
        raise AssertionError("a sealed funding row was stored")
    pg_store.finish_run(run_id, status="failed", counts=RunCounts(), error="x")
    assert scalar(pg_store, "SELECT count(*) FROM funding_rates") == 0


def _rows(store: Any, sql: str) -> list[Any]:
    with store.connection.cursor() as cursor:
        cursor.execute(sql)
        return list(cursor.fetchall())


def test_empty_or_failed_authorized_runs_change_nothing_permanent(
    postgres_url: str, pg_schema: str, pg_store: Any
) -> None:
    from trade_rl.integrations.market_store.store import PostgresMarketStore

    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 9, 1, tzinfo=UTC),
        now=datetime(2026, 10, 30, tzinfo=UTC),
        embargo=EmbargoPolicy.with_warmup(URL),
    )
    # an empty fetcher: every archive 404s; and a fetcher that fails the run outright
    empty = FakeFetcher()
    report = ingest(pg_store, empty, plan)
    assert report.counts.archives_missing == len(plan.specs) > 0
    broken = fetcher_for(plan)
    broken.fail(plan.specs[0].url, 503)
    with pytest.raises(Exception, match="503"):
        ingest(pg_store, broken, plan)
    # a brand-new session (no authorized run) is exactly as closed as before
    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as other:
        instrument_id, source_id = seed_instrument_and_source(other)
        import psycopg

        with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
            insert_funding(other, instrument_id, source_id, WARMUP)
        with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
            insert_funding(other, instrument_id, source_id, SEALED)
    assert scalar(pg_store, "SELECT count(*) FROM klines") == 0


def test_ingestion_enforces_the_boundaries_end_to_end(pg_store: Any) -> None:
    july = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=datetime(2026, 7, 1, tzinfo=UTC),
        end=FS1_WARMUP_START,
        now=datetime(2026, 12, 15, tzinfo=UTC),
    )
    report = ingest(pg_store, fetcher_for(july), july)
    assert report.counts.rows_inserted == 31 * 24
    assert scalar(pg_store, "SELECT max(open_time) FROM klines") == datetime(
        2026, 7, 31, 23, tzinfo=UTC
    )
    for end in (datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 11, 2, tzinfo=UTC)):
        with pytest.raises(SealedWindowEmbargoError):
            plan_ingestion(
                july.universe,
                start=datetime(2026, 7, 1, tzinfo=UTC),
                end=end,
                now=datetime(2026, 12, 15, tzinfo=UTC),
            )
    # a hand-built plan that smuggles an August archive in is refused with no download
    august = plan_ingestion(
        july.universe,
        start=datetime(2026, 8, 1, tzinfo=UTC),
        end=datetime(2026, 9, 1, tzinfo=UTC),
        now=datetime(2026, 12, 15, tzinfo=UTC),
        embargo=EmbargoPolicy.with_warmup(URL),
    )
    smuggled = IngestionPlan(
        august.universe,
        august.start,
        august.end,
        EmbargoPolicy.enforced(),
        august.specs,
        (),
    )
    fetcher = fetcher_for(august)
    with pytest.raises(Fs1WarmupSpanError):
        ingest(pg_store, fetcher, smuggled)
    assert fetcher.requested == []
    assert scalar(pg_store, "SELECT count(*) FROM klines") == 31 * 24


def test_authorized_warmup_ingestion_records_the_run_and_stays_below_the_sealed_window(
    pg_store: Any,
) -> None:
    policy = EmbargoPolicy.with_warmup(URL)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=datetime(2026, 9, 1, tzinfo=UTC),
        end=SEALED_WINDOW_START,
        now=datetime(2026, 12, 15, tzinfo=UTC),
        embargo=policy,
    )
    report = ingest(pg_store, fetcher_for(plan), plan)
    assert report.counts.rows_inserted == (30 + 31) * 24
    assert scalar(pg_store, "SELECT max(open_time) FROM klines") == datetime(
        2026, 10, 31, 23, tzinfo=UTC
    )
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT fs1_warmup_allowed, fs1_warmup_authorization_url, status, "
            "request ->> 'fs1_warmup_authorization_url' FROM ingestion_runs "
            "WHERE run_id = %s",
            (report.run_id,),
        )
        assert cursor.fetchone() == (True, URL, "succeeded", URL)
    with pytest.raises(SealedWindowEmbargoError):
        plan_ingestion(
            plan.universe,
            start=datetime(2026, 9, 1, tzinfo=UTC),
            end=datetime(2026, 11, 2, tzinfo=UTC),
            now=datetime(2027, 6, 1, tzinfo=UTC),
            embargo=policy,
        )


# -- hardening: run lifetime, session binding, immutability, TRUNCATE, provenance ----------


def start_authorized_run(store: Any) -> int:
    return int(
        store.begin_run(
            code_commit="c",
            command_line="c --allow-fs1-warmup-span",
            request={},
            embargo=EmbargoPolicy.with_warmup(URL),
        )
    )


def test_a_stale_authorized_run_stops_admitting_warmup_rows(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    run_id = start_authorized_run(pg_store)
    insert_funding(pg_store, instrument_id, source_id, WARMUP)
    # age the run beyond market_fs1_warmup_max_run_age() without touching its identity:
    # started_at is immutable, so emulate a hard-killed old run with a direct row
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO ingestion_runs (code_commit, command_line, request, "
            "started_at, fs1_warmup_allowed, fs1_warmup_authorization_url) "
            "VALUES ('c', 'c', '{}'::jsonb, now() - interval '13 hours', true, %s) "
            "RETURNING run_id",
            (URL,),
        )
        stale_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT set_config('trade_rl.fs1_warmup_run_id', %s, false)",
            (str(stale_id),),
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
        insert_funding(pg_store, instrument_id, source_id, "2026-09-01 00:00:00+00")
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)


def test_begin_run_reaps_abandoned_running_runs(pg_store: Any) -> None:
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO ingestion_runs (code_commit, command_line, request, "
            "started_at, fs1_warmup_allowed, fs1_warmup_authorization_url) "
            "VALUES ('c', 'c', '{}'::jsonb, now() - interval '30 hours', true, %s) "
            "RETURNING run_id",
            (URL,),
        )
        abandoned = cursor.fetchone()[0]
    run_id = pg_store.begin_run(
        code_commit="c", command_line="c", request={}, embargo=EmbargoPolicy.enforced()
    )
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT status, error, finished_at IS NOT NULL FROM ingestion_runs "
            "WHERE run_id = %s",
            (abandoned,),
        )
        status, error, finished = cursor.fetchone()
    assert status == "failed" and finished and "abandoned" in error
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)


def test_warmup_admission_is_bound_to_the_starting_session(
    postgres_url: str, pg_schema: str, pg_store: Any
) -> None:
    import psycopg

    from trade_rl.integrations.market_store.store import PostgresMarketStore

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    run_id = start_authorized_run(pg_store)
    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as other:
        with other.connection.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('trade_rl.fs1_warmup_run_id', %s, false)",
                (str(run_id),),
            )
        with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
            insert_funding(other, instrument_id, source_id, WARMUP)
    insert_funding(pg_store, instrument_id, source_id, WARMUP)
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)


def test_ingestion_runs_authorization_is_immutable_and_finished_runs_are_final(
    pg_store: Any,
) -> None:
    import psycopg

    run_id = start_authorized_run(pg_store)
    for sql in (
        "UPDATE ingestion_runs SET fs1_warmup_authorization_url = "
        "'https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1'",
        "UPDATE ingestion_runs SET fs1_warmup_allowed = false, "
        "fs1_warmup_authorization_url = NULL",
        "UPDATE ingestion_runs SET started_at = now()",
        "UPDATE ingestion_runs SET backend_pid = backend_pid + 1",
        "UPDATE ingestion_runs SET command_line = 'other'",
        "DELETE FROM ingestion_runs",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation):
            with pg_store.connection.cursor() as cursor:
                cursor.execute(sql)
    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)
    with pytest.raises(psycopg.errors.RestrictViolation, match="immutable"):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ingestion_runs SET status = 'running', finished_at = NULL "
                "WHERE run_id = %s",
                (run_id,),
            )
    # the reopened run (had it worked) would have re-admitted warm-up rows
    instrument_id, source_id = seed_instrument_and_source(pg_store)
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up"):
        insert_funding(pg_store, instrument_id, source_id, WARMUP)


def test_truncate_is_blocked_on_every_evidence_table_and_source_updates_are_constrained(
    pg_store: Any,
) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    insert_kline(pg_store, instrument_id, source_id)
    for sql in (
        "TRUNCATE klines",
        "TRUNCATE funding_rates",
        "TRUNCATE sources CASCADE",
        "TRUNCATE instruments CASCADE",
        "TRUNCATE ingestion_runs CASCADE",
        "TRUNCATE exchange_info_snapshots CASCADE",
        "TRUNCATE instrument_rule_snapshots",
        "TRUNCATE ingestion_conflicts",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation, match="TRUNCATE"):
            with pg_store.connection.cursor() as cursor:
                cursor.execute(sql)
    assert scalar(pg_store, "SELECT count(*) FROM klines") == 1

    # a 404 record may only gain evidence for the same archive identity
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
            "period_start, period_end, status, http_status, checksum_url, fetched_at, "
            "run_id) SELECT 'gone', 'klines', %s, '1h', 'monthly', '2024-02-01+00', "
            "'2024-03-01+00', 'missing', 404, 'gone.CHECKSUM', now(), run_id "
            "FROM ingestion_runs LIMIT 1",
            (instrument_id,),
        )
    for column, value in (
        ("period_end", "'2024-04-01+00'"),
        ("period_start", "'2024-01-15+00'"),
        ("url", "'other'"),
        ("bar_interval", "'4h'"),
    ):
        with pytest.raises(psycopg.errors.RestrictViolation, match="identity"):
            with pg_store.connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE sources SET {column} = {value} WHERE url = 'gone'"
                )


def test_export_receipt_records_warmup_runs_from_postgres(pg_store: Any) -> None:
    policy = EmbargoPolicy.with_warmup(URL)
    plan = plan_ingestion(
        Universe(symbols=SYMBOLS, markets=(StoreMarket.SPOT,)),
        start=datetime(2026, 7, 1, tzinfo=UTC),
        end=datetime(2026, 8, 3, tzinfo=UTC),
        now=datetime(2026, 10, 30, tzinfo=UTC),
        embargo=policy,
    )
    report = ingest(pg_store, fetcher_for(plan), plan)
    info = stored_exchange_info(StoreMarket.SPOT)
    pg_store.store_exchange_info(
        info,
        parse_symbol_rules(info.raw_payload, market=StoreMarket.SPOT, symbols=SYMBOLS),
        run_id=None,
    )
    export = export_market_dataset(
        pg_store,
        market=StoreMarket.SPOT,
        symbols=SYMBOLS,
        interval="1h",
        start_time=datetime(2026, 7, 1, tzinfo=UTC),
        end_time=datetime(2026, 8, 3, tzinfo=UTC),
        embargo=policy,
    )
    warmup = export.evidence["fs1_warmup"]
    assert warmup["touches_span"] is True
    assert warmup["runs"] == [{"authorization_url": URL, "run_id": report.run_id}]
    # an export that stays before the span records no warm-up runs
    before = export_market_dataset(
        pg_store,
        market=StoreMarket.SPOT,
        symbols=SYMBOLS,
        interval="1h",
        start_time=datetime(2026, 7, 1, tzinfo=UTC),
        end_time=datetime(2026, 8, 1, tzinfo=UTC),
    )
    assert before.evidence["fs1_warmup"]["touches_span"] is False
    assert before.evidence["fs1_warmup"]["runs"] == []
    assert before.dataset.dataset_id != export.dataset.dataset_id


# -- export ---------------------------------------------------------------------


@pytest.mark.parametrize("market", [StoreMarket.USDS_M, StoreMarket.SPOT])
def test_export_from_postgres_equals_the_file_backed_build(
    pg_store: Any, tmp_path: Path, market: StoreMarket
) -> None:
    plan = plan_for((market,))
    fetcher = fetcher_for(plan)
    gap_spec = next(s for s in plan.specs if s.symbol == "ETHUSDT" and s.interval)
    fetcher.add(
        gap_spec.url,
        archive_bytes(gap_spec, skip=frozenset({ms_from_utc(START) + 40 * HOUR_MS})),
    )
    ingest(pg_store, fetcher, plan)
    info = stored_exchange_info(market)
    rules = parse_symbol_rules(info.raw_payload, market=market, symbols=SYMBOLS)
    assert pg_store.store_exchange_info(info, rules, run_id=None) is True
    assert pg_store.store_exchange_info(info, rules, run_id=None) is False
    assert scalar(pg_store, "SELECT count(*) FROM instrument_rule_snapshots") == len(
        SYMBOLS
    )
    assert scalar(pg_store, "SELECT min(tick_size) FROM instrument_rule_snapshots") in (
        Decimal("0.10"),
        Decimal("0.01"),
    )

    export = export_market_dataset(
        pg_store,
        market=market,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
    )
    again = export_market_dataset(
        pg_store,
        market=market,
        symbols=SYMBOLS,
        interval="1h",
        start_time=START,
        end_time=END,
    )
    assert export.dataset.dataset_id == again.dataset.dataset_id

    file_build = file_backed_build(
        tmp_path,
        fetcher,
        plan,
        market=market,
        symbols=SYMBOLS,
        start=START,
        end=END,
        evidence=export.evidence,
    )
    assert export.dataset.dataset_id == file_build.dataset.dataset_id
    for name in ("close", "open", "high", "low", "volume", "funding_rate"):
        left = export.dataset.resolved_array(name)
        right = file_build.dataset.resolved_array(name)
        assert left.shape == right.shape and (left == right).all()


def test_export_refuses_the_window_and_status_reports(pg_store: Any) -> None:
    plan = plan_for((StoreMarket.SPOT,))
    ingest(pg_store, fetcher_for(plan), plan)
    with pytest.raises(SealedWindowEmbargoError):
        export_market_dataset(
            pg_store,
            market=StoreMarket.SPOT,
            symbols=SYMBOLS,
            interval="1h",
            start_time=START,
            end_time=datetime(2026, 11, 5, tzinfo=UTC),
        )
    status = pg_store.status()
    assert status["migrated"] is True
    assert status["embargo"]["sealed_from"].startswith("2026-11-01")  # type: ignore[index]
    assert "released" not in status["embargo"]  # type: ignore[operator]
    assert {item["symbol"] for item in status["klines"]} == set(SYMBOLS)  # type: ignore[attr-defined]
    assert status["sources"] == {"ingested": 2}
