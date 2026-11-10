from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    archive_bytes,
    fetcher_for,
)
from trade_rl.integrations.market_store.carry_source import (
    ArchiveDirFetcher,
    CarryCoverageError,
    VisionArchiveSource,
    hourly_quote_volume,
    load_carry_pair,
)
from trade_rl.integrations.market_store.checksum import ChecksumError
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    FundingRow,
    KlineRow,
    StoreMarket,
    Universe,
    ms_from_utc,
)
from trade_rl.integrations.market_store.ingest import FetchResult
from trade_rl.integrations.market_store.parse import ArchiveContentError
from trade_rl.integrations.market_store.plan import plan_ingestion

HOUR = 3_600_000
START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2022, 2, 1, tzinfo=UTC)


def january_plan(symbols=("SOLUSDT",)):
    return plan_ingestion(Universe(symbols=symbols), start=START, end=END, now=NOW)


# --- Vision archive source ----------------------------------------------------------


def test_source_verifies_each_archive_and_records_its_evidence() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    source = VisionArchiveSource(fetcher, plan)
    start_ms, end_ms = ms_from_utc(START), ms_from_utc(END)

    spot = source.read_klines(StoreMarket.SPOT, "SOLUSDT", "1h", start_ms, end_ms)
    perp = source.read_klines(StoreMarket.USDS_M, "SOLUSDT", "1h", start_ms, end_ms)
    funding = source.read_funding(StoreMarket.USDS_M, "SOLUSDT", start_ms, end_ms)

    assert len(spot) == len(perp) == 31 * 24
    assert [row.open_time_ms for row in spot] == list(range(start_ms, end_ms, HOUR))
    assert funding and all(start_ms <= row.funding_time_ms <= end_ms for row in funding)
    evidence = source.evidence()
    assert [item["url"] for item in evidence] == sorted(spec.url for spec in plan.specs)
    spec = next(s for s in plan.specs if s.kind is ArchiveKind.KLINES)
    entry = next(item for item in evidence if item["url"] == spec.url)
    import hashlib

    assert (
        entry["archive_sha256"] == hashlib.sha256(fetcher.files[spec.url]).hexdigest()
    )
    assert entry["checksum_url"] == f"{spec.url}.CHECKSUM"


def test_a_tampered_archive_is_a_checksum_error_not_a_coverage_gap() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.files[spec.url] = archive_bytes(spec, close_bump=1)
    source = VisionArchiveSource(fetcher, plan)
    with pytest.raises(ChecksumError):
        source.read_klines(
            StoreMarket.SPOT, "SOLUSDT", "1h", ms_from_utc(START), ms_from_utc(END)
        )
    assert not isinstance(ChecksumError("x"), CarryCoverageError)


def test_a_malformed_archive_is_not_swallowed_as_a_coverage_gap() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.add(spec.url, archive_bytes(spec, close_bump=10**9))
    with pytest.raises(Exception) as caught:
        VisionArchiveSource(fetcher, plan).read_klines(
            StoreMarket.SPOT, "SOLUSDT", "1h", ms_from_utc(START), ms_from_utc(END)
        )
    assert not isinstance(caught.value, CarryCoverageError)
    assert isinstance(caught.value, (ArchiveContentError, ValueError, OSError))


@pytest.mark.parametrize("defect", ["missing_archive", "missing_checksum", "http_500"])
def test_unavailable_archives_are_coverage_errors(defect: str) -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    if defect == "missing_archive":
        del fetcher.files[spec.url]
    elif defect == "missing_checksum":
        del fetcher.files[spec.checksum_url]
    else:
        fetcher.fail(spec.url, 500)
    with pytest.raises(CarryCoverageError):
        VisionArchiveSource(fetcher, plan).read_klines(
            StoreMarket.SPOT, "SOLUSDT", "1h", ms_from_utc(START), ms_from_utc(END)
        )


def test_unplanned_symbol_or_uncovered_period_is_a_coverage_error() -> None:
    plan = january_plan()
    source = VisionArchiveSource(fetcher_for(plan), plan)
    with pytest.raises(CarryCoverageError):
        source.read_klines(
            StoreMarket.SPOT, "ADAUSDT", "1h", ms_from_utc(START), ms_from_utc(END)
        )
    with pytest.raises(CarryCoverageError):
        source.read_klines(
            StoreMarket.SPOT,
            "SOLUSDT",
            "1h",
            ms_from_utc(START),
            ms_from_utc(datetime(2022, 3, 1, tzinfo=UTC)),
        )


def test_nothing_at_or_after_the_fs1_warmup_span_can_be_planned_or_read() -> None:
    from trade_rl.integrations.market_store.embargo import Fs1WarmupSpanError

    with pytest.raises(Fs1WarmupSpanError):
        plan_ingestion(
            Universe(symbols=("SOLUSDT",)),
            start=datetime(2026, 7, 1, tzinfo=UTC),
            end=datetime(2026, 8, 2, tzinfo=UTC),
            now=NOW,
        )


# --- on-disk archive cache ----------------------------------------------------------


class CountingFetcher(FakeFetcher):
    pass


def test_archive_dir_fetcher_is_offline_by_default_and_never_mistakes_a_miss_for_404(
    tmp_path,
) -> None:
    plan = january_plan()
    spec = plan.specs[0]
    result = ArchiveDirFetcher(tmp_path).fetch(spec.url)
    assert (result.status, result.body) == (0, None)


def test_archive_dir_fetcher_writes_once_and_serves_from_disk(tmp_path) -> None:
    plan = january_plan()
    inner = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    first = ArchiveDirFetcher(tmp_path, inner).fetch(spec.url)
    assert first.status == 200
    assert inner.requested == [spec.url]
    cached = ArchiveDirFetcher(tmp_path, FakeFetcher())  # inner has nothing
    second = cached.fetch(spec.url)
    assert second.status == 200 and second.body == first.body
    assert not list(tmp_path.rglob("*.part"))
    # spot and USD-M archives share a file name; the cache must keep them apart
    perp = next(s for s in plan.specs if s.market is StoreMarket.USDS_M)
    assert perp.filename == spec.filename or perp.kind is ArchiveKind.KLINES
    ArchiveDirFetcher(tmp_path, inner).fetch(perp.url)
    assert len(list(tmp_path.rglob("*.zip"))) == 2


def test_archive_dir_fetcher_does_not_cache_failures_and_refuses_odd_urls(
    tmp_path,
) -> None:
    plan = january_plan()
    spec = plan.specs[0]
    inner = FakeFetcher()
    inner.fail(spec.url, 404)
    assert ArchiveDirFetcher(tmp_path, inner).fetch(spec.url).status == 404
    assert not any(tmp_path.rglob("*.zip"))
    for url in (
        "https://example.com/data/spot/x.zip",
        "https://data.binance.vision/data/../x.zip",
    ):
        with pytest.raises(ValueError):
            ArchiveDirFetcher(tmp_path, inner).fetch(url)


def test_a_cached_archive_is_still_verified_against_its_checksum(tmp_path) -> None:
    plan = january_plan()
    inner = fetcher_for(plan)
    cache = ArchiveDirFetcher(tmp_path, inner)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    cache.fetch(spec.url)
    cache.fetch(spec.checksum_url)
    cached = next(tmp_path.rglob(spec.filename))
    cached.write_bytes(cached.read_bytes()[:-5] + b"XXXXX")
    with pytest.raises(Exception) as caught:
        VisionArchiveSource(ArchiveDirFetcher(tmp_path), plan).read_klines(
            StoreMarket.SPOT, "SOLUSDT", "1h", ms_from_utc(START), ms_from_utc(END)
        )
    assert not isinstance(caught.value, CarryCoverageError)


# --- pair loader ----------------------------------------------------------------------


def kline(open_ms: int, price: float = 100.0, quote: float = 1_000_000.0) -> KlineRow:
    value = Decimal(str(price))
    return KlineRow(
        open_time_ms=open_ms,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Decimal("10"),
        close_time_ms=open_ms + HOUR - 1,
        quote_volume=Decimal(str(quote)),
        trades=5,
        taker_buy_base_volume=Decimal("5"),
        taker_buy_quote_volume=Decimal("500"),
    )


class ListSource:
    def __init__(self, hours: int = 48, *, spot_skip=(), perp_skip=()) -> None:
        self.start = ms_from_utc(datetime(2023, 3, 24, tzinfo=UTC))
        self.hours = hours
        self.skip = {
            StoreMarket.SPOT: set(spot_skip),
            StoreMarket.USDS_M: set(perp_skip),
        }

    def read_klines(self, market, symbol, interval, start_ms, end_ms):
        return [
            kline(ms, price=100 + (ms // HOUR) % 3)
            for ms in range(start_ms, end_ms, HOUR)
            if (ms - self.start) // HOUR not in self.skip[market]
        ]

    def read_funding(self, market, symbol, start_ms, end_ms):
        return [
            FundingRow(funding_time_ms=ms, rate=Decimal("0.0001"))
            for ms in range(start_ms, end_ms + 1, 8 * HOUR)
            if start_ms <= ms <= end_ms
        ]


T0 = datetime(2023, 3, 24, tzinfo=UTC)
T1 = datetime(2023, 3, 26, tzinfo=UTC)
HALT = (
    (datetime(2023, 3, 24, 11, 27, tzinfo=UTC), datetime(2023, 3, 24, 14, tzinfo=UTC)),
)


def test_pair_loader_builds_the_unchanged_raw_series_contract() -> None:
    spot, perp = load_carry_pair(
        ListSource(), "SOLUSDT", start_time=T0, end_time=T1, halt_intervals=()
    )
    assert len(spot.timestamps) == len(perp.timestamps) == 48
    assert bool(spot.tradable.all()) and bool(perp.tradable.all())
    np.testing.assert_array_equal(spot.close, perp.close)
    assert not spot.funding_event_count.any()
    assert perp.funding_event_count.sum() == 6
    np.testing.assert_array_equal(perp.available_at, perp.timestamps)


def test_pair_loader_explains_only_declared_spot_halts() -> None:
    skipped = (12, 13)  # the bars opening 12:00 and 13:00, inside the halt
    with pytest.raises(CarryCoverageError, match="SOLUSDT"):
        load_carry_pair(
            ListSource(spot_skip=skipped),
            "SOLUSDT",
            start_time=T0,
            end_time=T1,
            halt_intervals=(),
        )
    spot, _ = load_carry_pair(
        ListSource(spot_skip=skipped),
        "SOLUSDT",
        start_time=T0,
        end_time=T1,
        halt_intervals=HALT,
    )
    assert not spot.tradable[11:14].any() and spot.tradable[14]
    assert spot.volume[12] == 0.0


def test_a_perpetual_gap_is_never_filled_even_inside_a_spot_halt() -> None:
    with pytest.raises(CarryCoverageError):
        load_carry_pair(
            ListSource(perp_skip=(12,)),
            "SOLUSDT",
            start_time=T0,
            end_time=T1,
            halt_intervals=HALT,
        )


def test_lookback_volume_requires_complete_hours_and_reads_only_quote_volume() -> None:
    values = hourly_quote_volume(
        ListSource(),
        StoreMarket.SPOT,
        "SOLUSDT",
        start_time=T0,
        end_time=T1,
    )
    assert values.shape == (48,) and set(values) == {1_000_000.0}
    with pytest.raises(CarryCoverageError, match="hourly bars"):
        hourly_quote_volume(
            ListSource(spot_skip=(5,)),
            StoreMarket.SPOT,
            "SOLUSDT",
            start_time=T0,
            end_time=T1,
        )


def test_fetch_result_type_is_the_ingestion_contract() -> None:
    assert FetchResult(status=200, body=b"x").status == 200
