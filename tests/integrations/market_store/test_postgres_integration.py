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
    SEALED_WINDOW_START,
    EmbargoPolicy,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.export import export_market_dataset
from trade_rl.integrations.market_store.ingest import (
    ConflictError,
    IngestReport,
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


# -- embargo at the database layer -----------------------------------------------


def test_database_rejects_sealed_rows_even_if_python_is_bypassed(pg_store: Any) -> None:
    import psycopg

    instrument_id, source_id = seed_instrument_and_source(pg_store)
    boundary = "2026-11-01 00:00:00+00"
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        insert_kline(
            pg_store,
            instrument_id,
            source_id,
            open_time=boundary,
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
    insert_kline(
        pg_store,
        instrument_id,
        source_id,
        open_time="2026-10-31 23:00+00",
        close_time="2026-10-31 23:59:59.999+00",
    )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO funding_rates (instrument_id, funding_time, rate, source_id) "
                "VALUES (%s, %s, 0.0001, %s)",
                (instrument_id, boundary, source_id),
            )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sources (url, kind, instrument_id, bar_interval, "
                "period_kind, period_start, period_end, status, http_status, "
                "checksum_url, fetched_at, run_id) SELECT 'm', 'klines', %s, '1h', "
                "'monthly', '2026-11-01+00', '2026-12-01+00', 'missing', 404, 'm.CHECKSUM', "
                "now(), run_id FROM ingestion_runs LIMIT 1",
                (instrument_id,),
            )


def test_embargo_row_cannot_be_edited_only_released_once(pg_store: Any) -> None:
    import psycopg

    state = pg_store.embargo_state()
    assert state.sealed_from == SEALED_WINDOW_START and not state.released
    for sql in (
        "UPDATE market_embargo SET sealed_from = '2030-01-01+00'",
        "UPDATE market_embargo SET reason = 'x'",
        "DELETE FROM market_embargo",
        "TRUNCATE market_embargo",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation):
            with pg_store.connection.cursor() as cursor:
                cursor.execute(sql)
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE market_embargo SET released_at = now(), "
                "release_authorization_url = 'https://example.com/x'"
            )
    with pytest.raises(psycopg.errors.CheckViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute("UPDATE market_embargo SET released_at = now()")


def test_ingestion_enforces_the_embargo_end_to_end(pg_store: Any) -> None:
    october = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=datetime(2026, 10, 1, tzinfo=UTC),
        end=SEALED_WINDOW_START,
        now=datetime(2026, 12, 15, tzinfo=UTC),
    )
    fetcher = fetcher_for(october)
    report = ingest(pg_store, fetcher, october)
    assert report.counts.rows_inserted == 31 * 24
    assert scalar(pg_store, "SELECT max(open_time) FROM klines") == datetime(
        2026, 10, 31, 23, tzinfo=UTC
    )
    with pytest.raises(SealedWindowEmbargoError):
        plan_ingestion(
            october.universe,
            start=datetime(2026, 10, 1, tzinfo=UTC),
            end=datetime(2026, 11, 2, tzinfo=UTC),
            now=datetime(2026, 12, 15, tzinfo=UTC),
        )
    assert not pg_store.embargo_state().released


def test_release_requires_the_authorization_url_and_is_permanent(pg_store: Any) -> None:
    run_id = pg_store.begin_run(
        code_commit="c",
        command_line="c --release-sealed-window",
        request={},
        embargo=EmbargoPolicy.release(URL),
    )
    from trade_rl.integrations.market_store.ingest import RunCounts

    pg_store.finish_run(run_id, status="succeeded", counts=RunCounts(), error=None)
    state = pg_store.embargo_state()
    assert state.released and state.authorization_url == URL
    assert (
        scalar(
            pg_store,
            "SELECT release_authorization_url FROM ingestion_runs WHERE run_id = %s",
            run_id,
        )
        == URL
    )
    instrument_id, source_id = seed_instrument_and_source(pg_store)
    insert_kline(
        pg_store,
        instrument_id,
        source_id,
        open_time="2026-11-01 00:00+00",
        close_time="2026-11-01 00:59:59.999+00",
    )
    import psycopg

    with pytest.raises(psycopg.errors.RestrictViolation):
        with pg_store.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE market_embargo SET released_at = NULL, release_authorization_url = NULL"
            )


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
    assert status["embargo"]["released"] is False  # type: ignore[index]
    assert {item["symbol"] for item in status["klines"]} == set(SYMBOLS)  # type: ignore[attr-defined]
    assert status["sources"] == {"ingested": 2}
