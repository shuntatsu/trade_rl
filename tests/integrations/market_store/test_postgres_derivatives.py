"""Real-PostgreSQL tests for the opt-in ``metrics`` / ``bookDepth`` tables (migration 0002).

Skipped unless ``TRADE_RL_TEST_POSTGRES_URL`` is set (CI's Market Store PostgreSQL job
sets it and fails on any skip). Each test uses its own throw-away schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    archive_bytes,
    fetcher_for,
)
from trade_rl.integrations.market_store.archive_url import (
    vision_daily_derivatives_url,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    BookDepthRow,
    OpenInterestMetricRow,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.ingest import (
    ConflictError,
    IngestReport,
    run_ingestion,
)
from trade_rl.integrations.market_store.parse import (
    parse_book_depth_archive,
    parse_open_interest_metrics_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-31"
DAY = datetime(2024, 3, 15, tzinfo=UTC)
NEW_KINDS = (ArchiveKind.METRICS, ArchiveKind.BOOK_DEPTH)
SEALED = "2026-11-01 00:00:00+00"
WARMUP = "2026-08-01 00:00:00+00"
METRIC_SQL = (
    "INSERT INTO open_interest_metrics (instrument_id, source_id, create_time, "
    "sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, "
    "sum_toptrader_long_short_ratio, count_long_short_ratio, "
    "sum_taker_long_short_vol_ratio) VALUES (%s, %s, %s, %s, 2, 1, 1, 1, 1)"
)
DEPTH_SQL = (
    "INSERT INTO book_depth (instrument_id, source_id, snapshot_time, percentage, "
    "depth, notional) VALUES (%s, %s, %s, %s, %s, 2)"
)


def plan_new(
    days: int = 2, *, symbols: tuple[str, ...] = ("BTCUSDT",)
) -> IngestionPlan:
    return plan_ingestion(
        Universe(symbols=symbols, markets=(StoreMarket.USDS_M,)),
        start=DAY,
        end=DAY + timedelta(days=days),
        now=NOW,
        kinds=NEW_KINDS,
    )


def fetcher_for_plan(plan: IngestionPlan, **kwargs: Any) -> FakeFetcher:
    fetcher = FakeFetcher()
    for spec in plan.specs:
        fetcher.add(spec.url, archive_bytes(spec, **kwargs))
    return fetcher


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


def execute(store: Any, sql: str, *params: Any) -> None:
    with store.connection.cursor() as cursor:
        # no parameters means the simple protocol, so a whole migration file can run
        cursor.execute(sql, params or None)


def seed(
    store: Any, *, kind: str = "metrics", market: StoreMarket = StoreMarket.USDS_M
) -> tuple[int, int]:
    """An instrument, a run and a one-day ``kind`` source to hang direct inserts on."""

    instrument_id = store.ensure_instrument(market, "BTCUSDT")
    run_id = scalar(
        store,
        "INSERT INTO ingestion_runs (code_commit, command_line, request) "
        "VALUES ('c', 'c', '{}'::jsonb) RETURNING run_id",
    )
    source_id = scalar(
        store,
        "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
        "period_start, period_end, status, http_status, sha256, size_bytes, "
        "checksum_url, checksum_file_sha256, checksum_line_sha256, fetched_at, "
        "row_count, run_id) VALUES (%s, %s, %s, NULL, 'daily', "
        "'2024-03-15+00', '2024-03-16+00', 'ingested', 200, %s, 1, 'u.CHECKSUM', "
        "%s, %s, now(), 1, %s) RETURNING source_id",
        f"u-{kind}",
        kind,
        instrument_id,
        "a" * 64,
        "b" * 64,
        "a" * 64,
        run_id,
    )
    return instrument_id, source_id


# -- migration ----------------------------------------------------------------------


def test_migration_0002_is_recorded_and_idempotent(pg_store: Any) -> None:
    from trade_rl.integrations.market_store.store import load_migrations

    shipped = load_migrations()
    assert shipped[2].name == "open_interest_book_depth"
    assert pg_store.schema_versions()[2] == (2, "open_interest_book_depth")
    assert (
        scalar(pg_store, "SELECT checksum_sha256 FROM schema_version WHERE version = 2")
        == shipped[2].checksum_sha256
    )
    assert pg_store.migrate() == []
    for table in ("open_interest_metrics", "book_depth"):
        assert scalar(pg_store, "SELECT to_regclass(%s) IS NOT NULL", table) is True


def test_migration_0002_upgrades_a_populated_0001_database(
    postgres_url: str, pg_schema: str
) -> None:
    from trade_rl.integrations.market_store.store import (
        PostgresMarketStore,
        load_migrations,
        migrate,
    )

    shipped = load_migrations()
    with PostgresMarketStore.open(postgres_url, schema=pg_schema) as store:
        for migration in shipped[:2]:
            with store.connection.transaction():
                execute(store, migration.sql)
                execute(
                    store,
                    "INSERT INTO schema_version (version, name, checksum_sha256) "
                    "VALUES (%s, %s, %s)",
                    migration.version,
                    migration.name,
                    migration.checksum_sha256,
                )
        instrument_id, source_id = seed(store, kind="funding_rate")
        assert instrument_id and source_id
        before = scalar(store, "SELECT count(*) FROM sources")
        applied = migrate(store.connection)
        assert [m.version for m in applied] == [2]
        assert scalar(store, "SELECT count(*) FROM sources") == before == 1
        assert scalar(store, "SELECT kind FROM sources") == "funding_rate"


# -- ingest -------------------------------------------------------------------------------


def test_ingest_is_idempotent_and_records_evidence(pg_store: Any) -> None:
    plan = plan_new()
    fetcher = fetcher_for_plan(plan)
    first = ingest(pg_store, fetcher, plan)
    assert first.counts.archives_ingested == len(plan.specs) == 4
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 2 * 288
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 2 * 1440
    assert first.counts.rows_inserted == 2 * 288 + 2 * 1440
    assert scalar(pg_store, "SELECT count(*) FROM klines") == 0
    assert scalar(pg_store, "SELECT count(*) FROM funding_rates") == 0

    fetcher.requested.clear()
    second = ingest(pg_store, fetcher, plan)
    assert fetcher.requested == []
    assert second.counts.archives_skipped == 4 and second.counts.rows_inserted == 0
    refetched = ingest(pg_store, fetcher, plan, refetch=True)
    assert (
        refetched.counts.archives_skipped == 4 and refetched.counts.rows_inserted == 0
    )
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 2 * 288
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 2 * 1440

    spec = next(s for s in plan.specs if s.kind is ArchiveKind.METRICS)
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT status, http_status, sha256, checksum_line_sha256, size_bytes, "
            "row_count, kind, bar_interval, period_kind, period_start, period_end "
            "FROM sources WHERE url = %s",
            (spec.url,),
        )
        status, http, sha, line, size, count, kind, interval, period, start, end = (
            cursor.fetchone()
        )
    payload = fetcher.files[spec.url]
    assert (status, http, kind, interval, period) == (
        "ingested",
        200,
        "metrics",
        None,
        "daily",
    )
    assert sha == line == hashlib.sha256(payload).hexdigest()
    assert size == len(payload) and count == 288 and end - start == timedelta(days=1)
    request = scalar(
        pg_store, "SELECT request FROM ingestion_runs WHERE run_id = %s", first.run_id
    )
    assert request["kinds"] == ["metrics", "book_depth"]
    assert len(request["archives"]) == 4


def test_a_second_symbol_and_a_missing_day_are_handled(pg_store: Any) -> None:
    plan = plan_new(symbols=("BTCUSDT", "ETHUSDT"))
    fetcher = fetcher_for_plan(plan)
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
    assert scalar(pg_store, "SELECT count(DISTINCT instrument_id) FROM book_depth") == 2


# -- conflicts ----------------------------------------------------------------------------


def test_a_changed_metric_is_refused_recorded_and_never_overwritten(
    pg_store: Any,
) -> None:
    plan = plan_new(days=1)
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.METRICS)
    parsed = parse_open_interest_metrics_archive(
        archive_bytes(spec), spec, embargo=EmbargoPolicy.enforced()
    )
    target = parsed[5]
    instrument_id, source_id = seed(pg_store)
    pg_store.insert_open_interest_metrics(
        instrument_id,
        source_id,
        [replace(target, sum_taker_long_short_vol_ratio=Decimal("9.99999999"))],
    )
    with pytest.raises(ConflictError, match="nothing was overwritten"):
        ingest(pg_store, fetcher_for_plan(plan), plan)
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 1
    assert scalar(
        pg_store,
        "SELECT sum_taker_long_short_vol_ratio FROM open_interest_metrics",
    ) == Decimal("9.99999999")
    assert (
        scalar(pg_store, "SELECT count(*) FROM sources WHERE url = %s", spec.url) == 0
    )
    with pg_store.connection.cursor() as cursor:
        cursor.execute(
            "SELECT kind, column_name, key_time, existing_value, incoming_value "
            "FROM ingestion_conflicts"
        )
        [(kind, column, key_time, existing, incoming)] = cursor.fetchall()
    assert (kind, column) == ("open_interest_metric", "sum_taker_long_short_vol_ratio")
    assert key_time == target.create_time
    assert Decimal(existing) == Decimal("9.99999999")
    assert Decimal(incoming) == target.sum_taker_long_short_vol_ratio
    assert (
        scalar(
            pg_store, "SELECT status FROM ingestion_runs ORDER BY run_id DESC LIMIT 1"
        )
        == "failed"
    )


def test_a_changed_band_is_refused_and_names_its_percentage(pg_store: Any) -> None:
    plan = plan_new(days=1)
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.BOOK_DEPTH)
    parsed = parse_book_depth_archive(
        archive_bytes(spec), spec, embargo=EmbargoPolicy.enforced()
    )
    target = parsed[3]
    instrument_id, source_id = seed(pg_store, kind="book_depth")
    pg_store.insert_book_depth(
        instrument_id, source_id, [replace(target, depth=target.depth + 1)]
    )
    with pytest.raises(ConflictError):
        ingest(pg_store, fetcher_for_plan(plan), plan)
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 1
    assert (
        scalar(
            pg_store,
            "SELECT column_name FROM ingestion_conflicts WHERE kind = 'book_depth'",
        )
        == f"depth@percentage={target.percentage}"
    )


def test_a_changed_archive_is_an_archive_conflict(pg_store: Any) -> None:
    plan = plan_new(days=1)
    ingest(pg_store, fetcher_for_plan(plan), plan)
    with pytest.raises(ConflictError) as caught:
        ingest(pg_store, fetcher_for_plan(plan, value_bump=1), plan, refetch=True)
    assert caught.value.conflicts[0].kind == "archive_sha256"
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 288
    assert (
        scalar(
            pg_store,
            "SELECT count(*) FROM ingestion_conflicts WHERE kind = 'archive_sha256'",
        )
        == 1
    )


# -- reading ------------------------------------------------------------------------------


def test_reads_are_exact_ordered_half_open_and_immutable(pg_store: Any) -> None:
    plan = plan_new()
    ingest(pg_store, fetcher_for_plan(plan), plan)
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.METRICS)
    expected = parse_open_interest_metrics_archive(
        archive_bytes(spec), spec, embargo=EmbargoPolicy.enforced()
    )

    metrics = pg_store.read_open_interest_metrics(
        "BTCUSDT", DAY, DAY + timedelta(days=1)
    )
    assert isinstance(metrics, tuple) and len(metrics) == 288
    assert all(isinstance(row, OpenInterestMetricRow) for row in metrics)
    assert [row.create_time_ms for row in metrics] == [
        r.create_time_ms for r in expected
    ]
    assert metrics[0].create_time == DAY
    assert metrics == tuple(expected)
    assert format(metrics[0].sum_open_interest, "f") == format(
        expected[0].sum_open_interest, "f"
    )
    assert format(metrics[0].sum_open_interest, "f") == "80000.0430000000000000"
    with pytest.raises(AttributeError):
        metrics[0].sum_open_interest = Decimal(0)  # type: ignore[misc]

    # half-open: the end instant is excluded, the start instant included
    window = pg_store.read_open_interest_metrics(
        "BTCUSDT", DAY + timedelta(minutes=5), DAY + timedelta(minutes=15)
    )
    assert [row.create_time for row in window] == [
        DAY + timedelta(minutes=5),
        DAY + timedelta(minutes=10),
    ]
    # a range spanning both days is contiguous and ascending
    both = pg_store.read_open_interest_metrics("BTCUSDT", DAY, DAY + timedelta(days=2))
    assert len(both) == 576
    assert [r.create_time_ms for r in both] == sorted(r.create_time_ms for r in both)
    assert (
        pg_store.read_open_interest_metrics("ETHUSDT", DAY, DAY + timedelta(days=2))
        == ()
    )
    # a non-UTC aware bound is converted, never reinterpreted
    plus9 = timezone(timedelta(hours=9))
    shifted = pg_store.read_open_interest_metrics(
        "BTCUSDT",
        datetime(2024, 3, 15, 9, 5, tzinfo=plus9),
        datetime(2024, 3, 15, 9, 15, tzinfo=plus9),
    )
    assert [row.create_time for row in shifted] == [r.create_time for r in window]

    depth_spec = next(s for s in plan.specs if s.kind is ArchiveKind.BOOK_DEPTH)
    expected_depth = parse_book_depth_archive(
        archive_bytes(depth_spec), depth_spec, embargo=EmbargoPolicy.enforced()
    )
    depth = pg_store.read_book_depth("BTCUSDT", DAY, DAY + timedelta(days=1))
    assert isinstance(depth, tuple) and len(depth) == 1440
    assert all(isinstance(row, BookDepthRow) for row in depth)
    assert depth == tuple(expected_depth)
    assert [(r.timestamp_ms, r.percentage) for r in depth] == sorted(
        (r.timestamp_ms, r.percentage) for r in depth
    )
    assert format(depth[0].depth, "f") == format(expected_depth[0].depth, "f")
    first_snapshot = depth[0].timestamp
    one_snapshot = pg_store.read_book_depth(
        "BTCUSDT", first_snapshot, first_snapshot + timedelta(seconds=1)
    )
    assert [row.percentage for row in one_snapshot] == [
        -5,
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
        5,
    ]
    with pytest.raises(AttributeError):
        depth[0].depth = Decimal(0)  # type: ignore[misc]


def test_reads_are_embargo_checked_and_symbol_checked(pg_store: Any) -> None:
    last = datetime(2026, 8, 1, tzinfo=UTC)
    assert (
        pg_store.read_open_interest_metrics("BTCUSDT", last - timedelta(days=1), last)
        == ()
    )
    assert pg_store.read_book_depth("BTCUSDT", last - timedelta(days=1), last) == ()
    for read in (pg_store.read_open_interest_metrics, pg_store.read_book_depth):
        with pytest.raises(Fs1WarmupSpanError):
            read("BTCUSDT", last - timedelta(days=1), last + timedelta(seconds=1))
        with pytest.raises(SealedWindowEmbargoError):
            read(
                "BTCUSDT",
                datetime(2026, 10, 31, tzinfo=UTC),
                datetime(2026, 11, 1, 0, 0, 1, tzinfo=UTC),
                embargo=EmbargoPolicy.with_warmup(AUTH),
            )
        with pytest.raises(ValueError, match="invalid symbol"):
            read("btc usdt", DAY, DAY + timedelta(days=1))
        with pytest.raises(ValueError, match="later than start"):
            read("BTCUSDT", DAY, DAY)


# -- append-only, constraints ----------------------------------------------------------------


def seeded_rows(store: Any) -> tuple[int, int, int]:
    instrument_id, metric_source = seed(store)
    _, depth_source = seed(store, kind="book_depth")
    execute(store, METRIC_SQL, instrument_id, metric_source, "2024-03-15 00:05+00", 5)
    execute(
        store, DEPTH_SQL, instrument_id, depth_source, "2024-03-15 00:00:09+00", 1, 7
    )
    return instrument_id, metric_source, depth_source


def test_rows_are_append_only_for_both_tables(pg_store: Any) -> None:
    import psycopg

    instrument_id, metric_source, depth_source = seeded_rows(pg_store)
    for sql in (
        "UPDATE open_interest_metrics SET sum_open_interest = 6",
        "DELETE FROM open_interest_metrics",
        "UPDATE book_depth SET depth = 8",
        "DELETE FROM book_depth",
    ):
        with pytest.raises(psycopg.errors.RestrictViolation, match="append-only"):
            execute(pg_store, sql)
    for table in ("open_interest_metrics", "book_depth"):
        with pytest.raises(psycopg.errors.RestrictViolation, match="TRUNCATE"):
            execute(pg_store, f"TRUNCATE {table}")
        with pytest.raises(psycopg.errors.RestrictViolation, match="TRUNCATE"):
            execute(pg_store, f"TRUNCATE {table} CASCADE")
    with pytest.raises(psycopg.errors.UniqueViolation):
        execute(
            pg_store, METRIC_SQL, instrument_id, metric_source, "2024-03-15 00:05+00", 9
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        execute(
            pg_store,
            DEPTH_SQL,
            instrument_id,
            depth_source,
            "2024-03-15 00:00:09+00",
            1,
            9,
        )
    # the same instant with another band is a different natural key
    execute(
        pg_store,
        DEPTH_SQL,
        instrument_id,
        depth_source,
        "2024-03-15 00:00:09+00",
        -1,
        9,
    )
    assert scalar(pg_store, "SELECT sum_open_interest FROM open_interest_metrics") == 5
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 2
    with pytest.raises(psycopg.errors.RestrictViolation, match="TRUNCATE"):
        execute(pg_store, "TRUNCATE sources CASCADE")


METRIC_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


@pytest.mark.parametrize("bad", ["-1", "'NaN'", "'Infinity'", "'-Infinity'"])
def test_constraints_reject_negative_nan_and_infinite_values(
    pg_store: Any, bad: str
) -> None:
    import psycopg

    instrument_id, metric_source = seed(pg_store)
    _, depth_source = seed(pg_store, kind="book_depth")
    for column in METRIC_COLUMNS:
        values = ", ".join(bad if name == column else "1" for name in METRIC_COLUMNS)
        with pytest.raises(psycopg.errors.CheckViolation):
            execute(
                pg_store,
                "INSERT INTO open_interest_metrics (instrument_id, source_id, "
                f"create_time, {', '.join(METRIC_COLUMNS)}) "
                f"VALUES (%s, %s, '2024-03-15 00:05+00', {values})",
                instrument_id,
                metric_source,
            )
    for depth, notional in ((bad, "1"), ("1", bad)):
        with pytest.raises(psycopg.errors.CheckViolation):
            execute(
                pg_store,
                "INSERT INTO book_depth (instrument_id, source_id, snapshot_time, "
                f"percentage, depth, notional) VALUES (%s, %s, "
                f"'2024-03-15 00:00:09+00', 1, {depth}, {notional})",
                instrument_id,
                depth_source,
            )
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 0
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 0


@pytest.mark.parametrize("percentage", [0, 101, -101])
def test_constraints_reject_an_impossible_percentage_band(
    pg_store: Any, percentage: int
) -> None:
    import psycopg

    instrument_id, depth_source = seed(pg_store, kind="book_depth")
    with pytest.raises(psycopg.errors.CheckViolation):
        execute(
            pg_store,
            DEPTH_SQL,
            instrument_id,
            depth_source,
            "2024-03-15 00:00:09+00",
            percentage,
            1,
        )


def test_zero_values_and_exact_decimals_are_stored_without_rounding(
    pg_store: Any,
) -> None:
    instrument_id, metric_source = seed(pg_store)
    execute(
        pg_store,
        "INSERT INTO open_interest_metrics (instrument_id, source_id, create_time, "
        "sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio, "
        "sum_toptrader_long_short_ratio, count_long_short_ratio, "
        "sum_taker_long_short_vol_ratio) VALUES (%s, %s, %s, "
        "84043.0430000000000000, 0, 0.00000001, 123456789012.1234567890123456, 0, 0)",
        instrument_id,
        metric_source,
        "2024-03-15 00:05+00",
    )
    [row] = pg_store.read_open_interest_metrics("BTCUSDT", DAY, DAY + timedelta(days=1))
    assert format(row.sum_open_interest, "f") == "84043.0430000000000000"
    assert format(row.sum_toptrader_long_short_ratio, "f") == (
        "123456789012.1234567890123456"
    )
    assert (
        row.sum_open_interest_value == 0
        and row.count_toptrader_long_short_ratio == Decimal("0.00000001")
    )


def test_only_usds_m_instruments_and_daily_sources_may_carry_the_new_kinds(
    pg_store: Any,
) -> None:
    import psycopg

    spot_id = pg_store.ensure_instrument(StoreMarket.SPOT, "BTCUSDT")
    usds_id, metric_source = seed(pg_store)
    _, depth_source = seed(pg_store, kind="book_depth")
    with pytest.raises(psycopg.errors.CheckViolation, match="USD-M"):
        execute(pg_store, METRIC_SQL, spot_id, metric_source, "2024-03-15 00:05+00", 5)
    with pytest.raises(psycopg.errors.CheckViolation, match="USD-M"):
        execute(
            pg_store, DEPTH_SQL, spot_id, depth_source, "2024-03-15 00:00:09+00", 1, 5
        )

    run_id = scalar(pg_store, "SELECT run_id FROM ingestion_runs LIMIT 1")
    source_sql = (
        "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
        "period_start, period_end, status, http_status, checksum_url, fetched_at, run_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'missing', 404, 'c', now(), %s)"
    )
    good = ("2024-03-20+00", "2024-03-21+00")
    cases = {
        "spot instrument": ("metrics", spot_id, None, "daily", *good),
        "monthly period": (
            "metrics",
            usds_id,
            None,
            "monthly",
            "2024-03-01+00",
            "2024-04-01+00",
        ),
        "interval set": ("book_depth", usds_id, "1h", "daily", *good),
        "two days": (
            "metrics",
            usds_id,
            None,
            "daily",
            "2024-03-20+00",
            "2024-03-22+00",
        ),
        "not midnight": (
            "book_depth",
            usds_id,
            None,
            "daily",
            "2024-03-20 06:00+00",
            "2024-03-21 06:00+00",
        ),
        "unknown kind": ("tickers", usds_id, None, "daily", *good),
    }
    for index, (label, case) in enumerate(cases.items()):
        with pytest.raises(psycopg.errors.CheckViolation):
            execute(pg_store, source_sql, f"x-{index}-{label}", *case, run_id)
    execute(
        pg_store, source_sql, "x-ok", "metrics", usds_id, None, "daily", *good, run_id
    )
    # the existing kinds keep their own shape
    execute(
        pg_store,
        source_sql,
        "x-kline",
        "klines",
        usds_id,
        "1h",
        "monthly",
        "2024-03-01+00",
        "2024-04-01+00",
        run_id,
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        execute(
            pg_store,
            source_sql,
            "x-kline-2",
            "klines",
            usds_id,
            None,
            "monthly",
            "2024-03-01+00",
            "2024-04-01+00",
            run_id,
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        execute(
            pg_store,
            "INSERT INTO ingestion_conflicts (run_id, kind, instrument_id, source_url, "
            "column_name) VALUES (%s, 'tickers', %s, 'u', 'c')",
            run_id,
            usds_id,
        )
    execute(
        pg_store,
        "INSERT INTO ingestion_conflicts (run_id, kind, instrument_id, source_url, "
        "column_name) VALUES (%s, 'book_depth', %s, 'u', 'c')",
        run_id,
        usds_id,
    )


# -- FS1 boundaries at the database layer ---------------------------------------------------------


def test_database_rejects_sealed_and_unauthorized_warmup_rows(pg_store: Any) -> None:
    import psycopg

    instrument_id, metric_source = seed(pg_store)
    _, depth_source = seed(pg_store, kind="book_depth")
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        execute(pg_store, METRIC_SQL, instrument_id, metric_source, SEALED, 1)
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        execute(pg_store, DEPTH_SQL, instrument_id, depth_source, SEALED, 1, 1)
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up span"):
        execute(pg_store, METRIC_SQL, instrument_id, metric_source, WARMUP, 1)
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up span"):
        execute(pg_store, DEPTH_SQL, instrument_id, depth_source, WARMUP, 1, 1)
    last = ("2026-07-31 23:55+00", "2026-07-31 23:59:59+00")
    execute(pg_store, METRIC_SQL, instrument_id, metric_source, last[0], 1)
    execute(pg_store, DEPTH_SQL, instrument_id, depth_source, last[1], 1, 1)
    # a source whose day reaches a closed span is refused as well
    run_id = scalar(pg_store, "SELECT run_id FROM ingestion_runs LIMIT 1")
    source_sql = (
        "INSERT INTO sources (url, kind, instrument_id, bar_interval, period_kind, "
        "period_start, period_end, status, http_status, checksum_url, fetched_at, run_id) "
        "VALUES (%s, %s, %s, NULL, 'daily', %s, %s, 'missing', 404, 'c', now(), %s)"
    )
    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up span"):
        execute(
            pg_store,
            source_sql,
            "w",
            "metrics",
            instrument_id,
            "2026-08-01+00",
            "2026-08-02+00",
            run_id,
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="sealed window"):
        execute(
            pg_store,
            source_sql,
            "s",
            "book_depth",
            instrument_id,
            "2026-11-01+00",
            "2026-11-02+00",
            run_id,
        )
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 1


def test_python_layers_refuse_embargoed_days_before_any_write(pg_store: Any) -> None:
    for kind in NEW_KINDS:
        for day, policy, error in (
            (
                datetime(2026, 8, 1, tzinfo=UTC),
                EmbargoPolicy.enforced(),
                Fs1WarmupSpanError,
            ),
            (
                datetime(2026, 11, 1, tzinfo=UTC),
                EmbargoPolicy.with_warmup(AUTH),
                SealedWindowEmbargoError,
            ),
        ):
            url = vision_daily_derivatives_url(kind, StoreMarket.USDS_M, "BTCUSDT", day)
            spec = ArchiveSpec(
                kind,
                StoreMarket.USDS_M,
                "BTCUSDT",
                None,
                PeriodKind.DAILY,
                day,
                day + timedelta(days=1),
                url,
                f"{url}.CHECKSUM",
            )
            plan = IngestionPlan(
                Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
                day,
                day + timedelta(days=1),
                policy,
                (spec,),
                (),
                (kind,),
            )
            fetcher = FakeFetcher()
            fetcher.add(url, archive_bytes(spec))
            with pytest.raises(error):
                ingest(pg_store, fetcher, plan)
            assert fetcher.requested == []
    assert scalar(pg_store, "SELECT count(*) FROM ingestion_runs") == 0
    assert scalar(pg_store, "SELECT count(*) FROM sources") == 0


def test_an_authorized_warmup_run_stores_and_reads_a_warmup_day(pg_store: Any) -> None:
    day = datetime(2026, 8, 1, tzinfo=UTC)
    policy = EmbargoPolicy.with_warmup(AUTH)
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=day,
        end=day + timedelta(days=1),
        now=NOW,
        embargo=policy,
        kinds=NEW_KINDS,
    )
    report = ingest(pg_store, fetcher_for_plan(plan), plan)
    assert report.counts.archives_ingested == 2
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 288
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 1440
    rows = pg_store.read_open_interest_metrics(
        "BTCUSDT", day, day + timedelta(days=1), embargo=policy
    )
    assert len(rows) == 288
    with pytest.raises(Fs1WarmupSpanError):
        pg_store.read_open_interest_metrics("BTCUSDT", day, day + timedelta(days=1))
    # outside an authorized run the same rows can still not be written
    instrument_id = pg_store.ensure_instrument(StoreMarket.USDS_M, "BTCUSDT")
    _, source_id = seed(pg_store)
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation, match="warm-up span"):
        execute(
            pg_store, METRIC_SQL, instrument_id, source_id, "2026-08-02 00:00+00", 1
        )


def test_default_ingest_does_not_touch_the_new_tables(pg_store: Any) -> None:
    plan = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M,)),
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        now=NOW,
    )
    report = ingest(pg_store, fetcher_for(plan), plan)
    assert report.counts.archives_ingested == len(plan.specs) == 2
    assert scalar(pg_store, "SELECT count(*) FROM open_interest_metrics") == 0
    assert scalar(pg_store, "SELECT count(*) FROM book_depth") == 0
    request = scalar(pg_store, "SELECT request FROM ingestion_runs LIMIT 1")
    assert "kinds" not in request
