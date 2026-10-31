from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    MemoryStore,
    archive_bytes,
    checksum_file,
    fetcher_for,
    kline_csv_rows,
    zip_csv,
)
from trade_rl.integrations.market_store.checksum import ChecksumError
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    FundingRow,
    KlineRow,
    PeriodKind,
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
from trade_rl.integrations.market_store.ingest import (
    ConflictError,
    IngestError,
    diff_rows,
    run_ingestion,
)
from trade_rl.integrations.market_store.parse import ArchiveContentError
from trade_rl.integrations.market_store.plan import IngestionPlan, plan_ingestion

URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-9"
UNIVERSE = Universe(
    symbols=("BTCUSDT",), markets=(StoreMarket.USDS_M, StoreMarket.SPOT)
)


def day(year: int, month: int, number: int = 1) -> datetime:
    return datetime(year, month, number, tzinfo=UTC)


def small_plan() -> IngestionPlan:
    return plan_ingestion(UNIVERSE, start=day(2024, 1), end=day(2024, 3), now=NOW)


def ingest(store: MemoryStore, fetcher: FakeFetcher, plan: IngestionPlan, **kwargs):  # type: ignore[no-untyped-def]
    return run_ingestion(
        store,
        fetcher,
        plan,
        code_commit="abc",
        command_line="test",
        clock=lambda: NOW,
        **kwargs,
    )


def test_ingest_stores_every_row_with_checksum_evidence() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    report = ingest(store, fetcher, plan)
    assert report.status == "succeeded"
    counts = report.counts
    assert counts.archives_planned == len(plan.specs) == 6
    assert counts.archives_ingested == 6
    assert counts.rows_inserted == (31 + 29) * 24 * 2 + (31 + 29) * 3
    assert counts.conflicts == counts.rows_identical == 0
    assert len(store.klines) == (31 + 29) * 24 * 2
    assert len(store.funding) == (31 + 29) * 3
    for spec in plan.specs:
        record = store.sources[spec.url]
        assert record["status"] == "ingested"
        assert (
            record["sha256"]
            == __import__("hashlib").sha256(fetcher.files[spec.url]).hexdigest()
        )
        assert record["checksum_file_sha256"] is not None
    assert store.runs[report.run_id]["status"] == "succeeded"


def test_reingest_is_a_no_op_without_touching_the_network() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    ingest(store, fetcher, plan)
    before = (dict(store.klines), dict(store.funding), dict(store.sources))
    fetcher.requested.clear()
    report = ingest(store, fetcher, plan)
    assert fetcher.requested == []
    assert report.counts.archives_skipped == 6
    assert report.counts.archives_ingested == report.counts.rows_inserted == 0
    assert (store.klines, store.funding, store.sources) == before


def test_refetching_identical_bytes_is_also_a_no_op() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    ingest(store, fetcher, plan)
    before = dict(store.klines)
    report = ingest(store, fetcher, plan, refetch=True)
    assert len(fetcher.requested) == 2 * 6 + 2 * 6
    assert report.counts.archives_skipped == 6
    assert store.klines == before


def test_overlapping_daily_archive_adds_nothing_new() -> None:
    store = MemoryStore()
    monthly = plan_ingestion(
        Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,)),
        start=day(2024, 1),
        end=day(2024, 2),
        now=NOW,
    )
    ingest(store, fetcher_for(monthly), monthly)
    daily_spec = ArchiveSpec(
        kind=ArchiveKind.KLINES,
        market=StoreMarket.SPOT,
        symbol="BTCUSDT",
        interval="1h",
        period_kind=PeriodKind.DAILY,
        period_start=day(2024, 1, 5),
        period_end=day(2024, 1, 6),
        url="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-05.zip",
        checksum_url="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-05.zip.CHECKSUM",
    )
    fetcher = FakeFetcher()
    month_rows = kline_csv_rows(
        "BTCUSDT", ms_from_utc(day(2024, 1)), ms_from_utc(day(2024, 2))
    )
    day_rows = [
        row
        for row in month_rows
        if ms_from_utc(day(2024, 1, 5)) <= int(row[0]) < ms_from_utc(day(2024, 1, 6))
    ]
    fetcher.add(daily_spec.url, zip_csv(day_rows))
    plan = IngestionPlan(
        UNIVERSE,
        day(2024, 1, 5),
        day(2024, 1, 6),
        EmbargoPolicy.enforced(),
        (daily_spec,),
        (),
    )
    report = ingest(store, fetcher, plan)
    assert report.counts.rows_inserted == 0
    assert report.counts.rows_identical == 24


def test_changed_value_for_an_existing_key_is_refused_and_recorded() -> None:
    plan, store = small_plan(), MemoryStore()
    ingest(store, fetcher_for(plan), plan)
    snapshot = (dict(store.klines), dict(store.funding), dict(store.sources))

    first = plan.specs[0]
    changed = FakeFetcher()
    changed.add(
        first.url,
        zip_csv(
            kline_csv_rows(
                first.symbol,
                ms_from_utc(first.period_start),
                ms_from_utc(first.period_end),
                close_bump=1,
            )
        ),
    )
    # a different archive hash under the same URL, with a valid checksum, is a conflict
    # of the archive itself when refetching ...
    with pytest.raises(ConflictError) as excinfo:
        ingest(store, changed, plan, refetch=True)
    assert excinfo.value.conflicts[0].kind == "archive_sha256"
    assert (store.klines, store.funding, store.sources) == snapshot
    assert store.runs[max(store.runs)]["status"] == "failed"

    # ... and a changed value under a *new* source (the same keys, another archive
    # URL) is a per-row conflict that rolls the archive back
    other_url = first.url.replace("monthly", "daily").replace(
        "2024-01.zip", "2024-01-01.zip"
    )
    other = ArchiveSpec(
        kind=first.kind,
        market=first.market,
        symbol=first.symbol,
        interval=first.interval,
        period_kind=PeriodKind.DAILY,
        period_start=first.period_start,
        period_end=day(2024, 1, 2),
        url=other_url,
        checksum_url=f"{other_url}.CHECKSUM",
    )
    daily = FakeFetcher()
    daily.add(
        other.url,
        zip_csv(
            kline_csv_rows(
                other.symbol,
                ms_from_utc(other.period_start),
                ms_from_utc(other.period_end),
                close_bump=1,
            )
        ),
    )
    conflict_plan = IngestionPlan(
        UNIVERSE, day(2024, 1), day(2024, 1, 2), EmbargoPolicy.enforced(), (other,), ()
    )
    with pytest.raises(ConflictError) as row_excinfo:
        ingest(store, daily, conflict_plan)
    columns = {item.column for item in row_excinfo.value.conflicts}
    assert "close" in columns
    assert all(item.kind == "kline" for item in row_excinfo.value.conflicts)
    assert (store.klines, store.funding, store.sources) == snapshot
    assert store.conflicts[-1].existing != store.conflicts[-1].incoming
    assert other.url not in store.sources  # the refused archive left no evidence behind


def test_checksum_mismatch_fails_closed_before_any_write() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    victim = plan.specs[0]
    fetcher.add(
        victim.url,
        fetcher.files[victim.url],
        checksum=checksum_file(b"something else", victim.filename),
    )
    with pytest.raises(ChecksumError):
        ingest(store, fetcher, plan)
    assert store.klines == {} and store.funding == {} and store.sources == {}
    assert store.runs[1]["status"] == "failed"


def test_missing_checksum_file_fails_closed() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    del fetcher.files[f"{plan.specs[0].url}.CHECKSUM"]
    with pytest.raises(IngestError, match="checksum unavailable"):
        ingest(store, fetcher, plan)
    assert store.klines == {}


def test_a_404_is_recorded_as_missing_and_retried_next_run() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    gone = plan.specs[1]
    del fetcher.files[gone.url]
    report = ingest(store, fetcher, plan)
    assert report.counts.archives_missing == 1
    assert report.missing_urls == (gone.url,)
    assert store.sources[gone.url]["status"] == "missing"
    # the archive is published later: the next run ingests it
    fetcher.add(gone.url, archive_bytes(gone))
    again = ingest(store, fetcher, plan)
    assert again.counts.archives_ingested == 1
    assert again.counts.archives_skipped == 5
    assert store.sources[gone.url]["status"] == "ingested"


def test_a_vanished_ingested_archive_is_an_error_on_refetch() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    ingest(store, fetcher, plan)
    del fetcher.files[plan.specs[0].url]
    with pytest.raises(IngestError, match="now missing"):
        ingest(store, fetcher, plan, refetch=True)


def test_server_errors_abort_the_run() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    fetcher.fail(plan.specs[0].url, 503)
    with pytest.raises(IngestError, match="HTTP 503"):
        ingest(store, fetcher, plan)
    assert store.runs[1]["status"] == "failed"


def test_corrupt_archive_with_valid_checksum_is_rejected() -> None:
    plan, store = small_plan(), MemoryStore()
    fetcher = fetcher_for(plan)
    spec = plan.specs[0]
    fetcher.add(spec.url, b"this is not a zip archive")
    with pytest.raises(ArchiveContentError):
        ingest(store, fetcher, plan)
    assert store.klines == {}


# -- embargo through the ingestion path ------------------------------------------


def month_spec(
    year: int, month: int, *, daily_end: datetime | None = None
) -> ArchiveSpec:
    start = day(year, month)
    end = day(year + (month == 12), month % 12 + 1)
    url = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
        f"BTCUSDT-1h-{year}-{month:02d}.zip"
    )
    return ArchiveSpec(
        kind=ArchiveKind.KLINES,
        market=StoreMarket.SPOT,
        symbol="BTCUSDT",
        interval="1h",
        period_kind=PeriodKind.MONTHLY,
        period_start=start,
        period_end=end if daily_end is None else daily_end,
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


def one_archive_plan(spec: ArchiveSpec, policy: EmbargoPolicy) -> IngestionPlan:
    return IngestionPlan(
        UNIVERSE, spec.period_start, spec.period_end, policy, (spec,), ()
    )


POLICIES = [EmbargoPolicy.enforced(), EmbargoPolicy.with_warmup(URL)]
POLICY_IDS = ["default", "warmup-authorized"]


@pytest.mark.parametrize("policy", POLICIES, ids=POLICY_IDS)
def test_a_sealed_window_archive_is_never_downloaded_under_any_policy(
    policy: EmbargoPolicy,
) -> None:
    spec = month_spec(2026, 11)
    fetcher, store = FakeFetcher(), MemoryStore()
    fetcher.add(spec.url, archive_bytes(spec))
    with pytest.raises(SealedWindowEmbargoError):
        ingest(store, fetcher, one_archive_plan(spec, policy))
    assert fetcher.requested == []
    assert store.klines == {} and store.sources == {}


def test_a_warmup_archive_is_never_downloaded_by_default() -> None:
    fetcher, store = FakeFetcher(), MemoryStore()
    for month in (8, 9, 10):
        spec = month_spec(2026, month)
        fetcher.add(spec.url, archive_bytes(spec))
        with pytest.raises(Fs1WarmupSpanError):
            ingest(store, fetcher, one_archive_plan(spec, EmbargoPolicy.enforced()))
    assert fetcher.requested == []
    assert store.klines == {} and store.sources == {}
    assert all(run["status"] == "failed" for run in store.runs.values())


def test_a_late_row_in_an_allowed_archive_is_refused_before_any_write() -> None:
    # July (allowed by default) carrying one August bar
    july = month_spec(2026, 7)
    fetcher, store = FakeFetcher(), MemoryStore()
    fetcher.add(
        july.url,
        zip_csv(
            kline_csv_rows(
                "BTCUSDT",
                ms_from_utc(july.period_start),
                ms_from_utc(day(2026, 8)) + 3_600_000,
            )
        ),
    )
    with pytest.raises(Fs1WarmupSpanError):
        ingest(store, fetcher, one_archive_plan(july, EmbargoPolicy.enforced()))
    assert store.klines == {} and store.sources == {}

    # October (allowed only with the warm-up authorization) carrying one November bar
    october = month_spec(2026, 10)
    fetcher.add(
        october.url,
        zip_csv(
            kline_csv_rows(
                "BTCUSDT",
                ms_from_utc(october.period_start),
                ms_from_utc(SEALED_WINDOW_START) + 3_600_000,
            )
        ),
    )
    with pytest.raises(SealedWindowEmbargoError):
        ingest(
            store, fetcher, one_archive_plan(october, EmbargoPolicy.with_warmup(URL))
        )
    assert store.klines == {} and store.sources == {}


def test_july_archive_ending_at_the_default_boundary_ingests_fully() -> None:
    spec = month_spec(2026, 7)
    fetcher, store = FakeFetcher(), MemoryStore()
    fetcher.add(spec.url, archive_bytes(spec))
    report = ingest(store, fetcher, one_archive_plan(spec, EmbargoPolicy.enforced()))
    assert report.counts.rows_inserted == 31 * 24
    last = max(key[2] for key in store.klines)
    assert last == ms_from_utc(FS1_WARMUP_START) - 3_600_000
    assert store.runs[report.run_id]["fs1_warmup_allowed"] is False


def test_warmup_authorization_is_recorded_and_admits_only_the_warmup_span() -> None:
    policy = EmbargoPolicy.with_warmup(URL)
    store, fetcher = MemoryStore(), FakeFetcher()
    specs = tuple(month_spec(2026, month) for month in (8, 9, 10))
    for spec in specs:
        fetcher.add(spec.url, archive_bytes(spec))
    plan = IngestionPlan(UNIVERSE, day(2026, 8), SEALED_WINDOW_START, policy, specs, ())
    report = ingest(store, fetcher, plan)
    assert report.counts.rows_inserted == (31 + 30 + 31) * 24
    run = store.runs[report.run_id]
    assert run["fs1_warmup_allowed"] is True
    assert run["fs1_warmup_authorization_url"] == URL
    assert (
        max(key[2] for key in store.klines)
        == ms_from_utc(SEALED_WINDOW_START) - 3_600_000
    )


# -- diff -----------------------------------------------------------------------


def row(close: str = "1.5") -> KlineRow:
    return KlineRow(
        0,
        Decimal("1"),
        Decimal("2"),
        Decimal("1"),
        Decimal(close),
        Decimal("3"),
        9,
        Decimal("4"),
        5,
        Decimal("6"),
        Decimal("7"),
    )


def test_diff_classifies_new_identical_and_changed_rows() -> None:
    existing = {0: row("1.5"), 1: row("1.5")}
    incoming = [(0, row("1.50")), (1, row("1.6")), (2, row("1.5"))]
    result = diff_rows(existing, incoming)
    assert result.identical == 1
    assert [key for key, _ in result.new] == [2]
    assert result.conflicts == ((1, "close", "1.5", "1.6"),)


def test_funding_diff_compares_every_field() -> None:
    stored = {1: FundingRow(1, Decimal("0.0001"), None, 8)}
    changed = diff_rows(stored, [(1, FundingRow(1, Decimal("0.0001"), None, 4))])
    assert changed.conflicts == ((1, "funding_interval_hours", "8", "4"),)
    same = diff_rows(stored, [(1, FundingRow(1, Decimal("0.00010"), None, 8))])
    assert same.identical == 1 and not same.conflicts
