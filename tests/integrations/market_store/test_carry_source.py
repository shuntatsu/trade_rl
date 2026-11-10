import errno
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from threading import Barrier

import numpy as np
import pytest

from tests.integrations.market_store.support import (
    NOW,
    FakeFetcher,
    archive_bytes,
    fetcher_for,
)
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryOperationalHalt,
)
from trade_rl.integrations.market_store.carry_source import (
    OFFLINE_MISS_STATUS,
    ArchiveDirFetcher,
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


def read_spot(source, symbol="SOLUSDT"):
    return source.read_klines(
        StoreMarket.SPOT, symbol, "1h", ms_from_utc(START), ms_from_utc(END)
    )


def no_sleep(_seconds: float) -> None:
    return None


def test_a_tampered_archive_is_an_integrity_halt_never_a_coverage_gap() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.files[spec.url] = archive_bytes(spec, close_bump=1)
    with pytest.raises(CarryIntegrityHalt) as caught:
        read_spot(VisionArchiveSource(fetcher, plan))
    assert not isinstance(caught.value, CarryCoverageError)
    assert caught.value.evidence["url"] == spec.url
    assert caught.value.evidence["kind"] == "checksum_mismatch"
    assert isinstance(caught.value.__cause__, ChecksumError)


def test_a_malformed_archive_is_an_integrity_halt() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.add(spec.url, archive_bytes(spec, close_bump=10**9))
    with pytest.raises(CarryIntegrityHalt) as caught:
        read_spot(VisionArchiveSource(fetcher, plan))
    assert caught.value.evidence["kind"] == "malformed_archive"
    assert isinstance(caught.value.__cause__, ArchiveContentError)


def test_an_archive_404_is_the_only_archive_level_coverage_fact() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    del fetcher.files[spec.url]
    with pytest.raises(CarryCoverageError) as caught:
        read_spot(VisionArchiveSource(fetcher, plan))
    assert caught.value.evidence == {
        "kind": "archive_not_published",
        "url": spec.url,
        "status": 404,
    }


def test_a_missing_checksum_file_cannot_prove_the_archive_and_halts() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    del fetcher.files[spec.checksum_url]
    with pytest.raises(CarryIntegrityHalt) as caught:
        read_spot(VisionArchiveSource(fetcher, plan))
    assert caught.value.evidence["url"] == spec.checksum_url
    assert caught.value.evidence["status"] == 404


@pytest.mark.parametrize("status", [500, 502, 503, 429, 0])
def test_persistent_transient_failures_halt_after_a_bounded_retry(status) -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.fail(spec.url, status)
    source = VisionArchiveSource(fetcher, plan, max_attempts=3, sleep=no_sleep)
    with pytest.raises(CarryOperationalHalt) as caught:
        read_spot(source)
    assert not isinstance(caught.value, CarryCoverageError)
    assert caught.value.resumable is True
    assert caught.value.evidence == {
        "kind": "archive_unavailable",
        "url": spec.url,
        "status": status,
        "attempts": 3,
    }
    assert fetcher.requested.count(spec.url) == 3


def test_a_transient_failure_that_clears_is_retried_to_success() -> None:
    plan = january_plan()
    inner = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)

    class Flaky:
        def __init__(self) -> None:
            self.failures = 2

        def fetch(self, url):
            if url == spec.url and self.failures:
                self.failures -= 1
                return FetchResult(status=503, body=None)
            return inner.fetch(url)

    sleeps: list[float] = []
    source = VisionArchiveSource(Flaky(), plan, max_attempts=3, sleep=sleeps.append)
    assert len(read_spot(source)) == 31 * 24
    assert len(sleeps) == 2 and sleeps == sorted(sleeps)


def test_four_xx_other_than_404_is_not_retried_and_halts() -> None:
    plan = january_plan()
    fetcher = fetcher_for(plan)
    spec = next(s for s in plan.specs if s.market is StoreMarket.SPOT)
    fetcher.fail(spec.url, 403)
    with pytest.raises(CarryOperationalHalt):
        read_spot(VisionArchiveSource(fetcher, plan, sleep=no_sleep))
    assert fetcher.requested.count(spec.url) == 1


def test_an_offline_cache_miss_halts_and_is_never_a_coverage_fact(tmp_path) -> None:
    plan = january_plan()
    source = VisionArchiveSource(ArchiveDirFetcher(tmp_path), plan, sleep=no_sleep)
    with pytest.raises(CarryOperationalHalt) as caught:
        read_spot(source)
    assert not isinstance(caught.value, CarryCoverageError)
    assert caught.value.evidence["kind"] == "offline_cache_miss"
    assert caught.value.evidence["status"] == OFFLINE_MISS_STATUS


def test_unplanned_symbol_or_uncovered_period_is_a_plan_defect_that_halts() -> None:
    plan = january_plan()
    source = VisionArchiveSource(fetcher_for(plan), plan)
    with pytest.raises(CarryIntegrityHalt):
        read_spot(source, "ADAUSDT")
    with pytest.raises(CarryIntegrityHalt):
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
    assert (result.status, result.body) == (OFFLINE_MISS_STATUS, None)


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


def test_archive_cache_late_rival_is_not_overwritten(tmp_path) -> None:
    """A competing on-disk file wins even if it appeared inside inner.fetch."""
    spec = january_plan().specs[0]
    target = ArchiveDirFetcher(tmp_path)._path(spec.url)
    winner = b"concurrent publisher bytes"

    class RacingFetcher:
        def fetch(self, url: str) -> FetchResult:
            assert url == spec.url
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(winner)
            return FetchResult(status=200, body=b"newly downloaded bytes")

    actual = ArchiveDirFetcher(tmp_path, RacingFetcher()).fetch(spec.url)
    assert actual.status == 200
    assert actual.body == winner
    assert target.read_bytes() == winner
    assert not list(tmp_path.rglob("*.part-*"))


def test_archive_cache_concurrent_publishers_have_private_stages(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both fetchers must return the one published winner and leak no temp file."""
    spec = january_plan().specs[0]
    target = ArchiveDirFetcher(tmp_path)._path(spec.url)
    real_link = os.link
    at_link = Barrier(2)
    stages = []

    def racing_link(staging, destination):
        assert destination == target
        stages.append(staging)
        at_link.wait(timeout=10)
        return real_link(staging, destination)

    class StaticFetcher:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def fetch(self, url: str) -> FetchResult:
            assert url == spec.url
            return FetchResult(status=200, body=self.payload)

    monkeypatch.setattr(os, "link", racing_link)
    fetchers = (
        ArchiveDirFetcher(tmp_path, StaticFetcher(b"first archive")),
        ArchiveDirFetcher(tmp_path, StaticFetcher(b"second archive")),
    )
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(workers.map(lambda cache: cache.fetch(spec.url), fetchers))
    winner = target.read_bytes()
    assert winner in {b"first archive", b"second archive"}
    assert len(stages) == len(set(stages)) == 2
    assert all(result.status == 200 and result.body == winner for result in results)
    assert not list(tmp_path.rglob("*.part-*"))


def test_archive_cache_unsupported_exclusive_publish_halts_without_partial_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lack of hard-link support is operational failure, never silent fallback."""
    spec = january_plan().specs[0]
    target = ArchiveDirFetcher(tmp_path)._path(spec.url)

    class StaticFetcher:
        def fetch(self, _url: str) -> FetchResult:
            return FetchResult(status=200, body=b"readable download")

    def unsupported(_src, _dst) -> None:
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)
    with pytest.raises(CarryOperationalHalt, match="cache"):
        ArchiveDirFetcher(tmp_path, StaticFetcher()).fetch(spec.url)
    assert not target.exists()
    assert not list(tmp_path.rglob("*.part-*"))


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
    with pytest.raises(CarryIntegrityHalt):
        read_spot(VisionArchiveSource(ArchiveDirFetcher(tmp_path), plan))


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
    with pytest.raises(CarryCoverageError, match="missing hourly bar"):
        hourly_quote_volume(
            ListSource(spot_skip=(5,)),
            StoreMarket.SPOT,
            "SOLUSDT",
            start_time=T0,
            end_time=T1,
        )


def test_fetch_result_type_is_the_ingestion_contract() -> None:
    assert FetchResult(status=200, body=b"x").status == 200


# --- row-level integrity versus coverage in the pair loader -------------------------


class RowsSource(ListSource):
    """ListSource with row-level defects injected after generation."""

    def __init__(self, *, spot=None, perp=None, funding=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spot, self.perp, self.fund = spot, perp, funding

    def read_klines(self, market, symbol, interval, start_ms, end_ms):
        rows = super().read_klines(market, symbol, interval, start_ms, end_ms)
        edit = self.spot if market is StoreMarket.SPOT else self.perp
        return edit(rows) if edit else rows

    def read_funding(self, market, symbol, start_ms, end_ms):
        rows = super().read_funding(market, symbol, start_ms, end_ms)
        return self.fund(rows) if self.fund else rows


def load(source, halts=()):
    return load_carry_pair(
        source, "SOLUSDT", start_time=T0, end_time=T1, halt_intervals=halts
    )


@pytest.mark.parametrize("leg", ["spot", "perp"])
@pytest.mark.parametrize("defect", ["duplicate", "unordered"])
def test_duplicate_or_unordered_rows_are_integrity_halts(leg, defect) -> None:
    def edit(rows):
        if defect == "duplicate":
            return [*rows[:5], rows[4], *rows[5:]]
        return [*rows[:5], rows[6], rows[5], *rows[7:]]

    with pytest.raises(CarryIntegrityHalt) as caught:
        load(RowsSource(**{leg: edit}))
    assert not isinstance(caught.value, CarryCoverageError)
    assert caught.value.evidence["kind"] == f"{defect}_rows"
    assert caught.value.evidence["market"] == ("spot" if leg == "spot" else "usds_m")


def test_duplicate_funding_events_are_an_integrity_halt() -> None:
    with pytest.raises(CarryIntegrityHalt):
        load(RowsSource(funding=lambda rows: [*rows, rows[0]]))


def test_a_missing_bar_is_a_coverage_fact_with_its_evidence() -> None:
    with pytest.raises(CarryCoverageError) as caught:
        load(ListSource(spot_skip=(12,)))
    evidence = caught.value.evidence
    assert evidence["kind"] == "gap" and evidence["market"] == "spot"
    assert evidence["missing_hours"] == 1
    assert evidence["first_missing_open_ms"] == ms_from_utc(
        datetime(2023, 3, 24, 12, tzinfo=UTC)
    )


def test_a_gap_only_partly_inside_a_declared_halt_is_still_a_coverage_fact() -> None:
    with pytest.raises(CarryCoverageError):
        load(ListSource(spot_skip=(10, 11, 12)), halts=HALT)  # 10:00 is not halted


def test_a_missing_first_bar_cannot_be_explained_by_a_halt() -> None:
    with pytest.raises(CarryCoverageError):
        load(
            ListSource(spot_skip=(0,)),
            halts=((T0, datetime(2023, 3, 24, 2, tzinfo=UTC)),),
        )


def test_a_funding_gap_is_a_coverage_fact() -> None:
    with pytest.raises(CarryCoverageError) as caught:
        load(RowsSource(funding=lambda rows: rows[:2]))
    assert caught.value.evidence["kind"] == "funding_gap"


def test_lookback_rows_follow_the_same_split() -> None:
    kwargs = dict(start_time=T0, end_time=T1)
    with pytest.raises(CarryIntegrityHalt):
        hourly_quote_volume(
            RowsSource(spot=lambda rows: [*rows[:3], rows[2], *rows[3:]]),
            StoreMarket.SPOT,
            "SOLUSDT",
            **kwargs,
        )
    with pytest.raises(CarryCoverageError):
        hourly_quote_volume(
            ListSource(spot_skip=(7,)), StoreMarket.SPOT, "SOLUSDT", **kwargs
        )


def test_funding_gap_rule_matches_the_assemblers_rule() -> None:
    from dataclasses import replace

    from tests.integrations.test_binance_carry import series
    from trade_rl.integrations.binance.carry import assemble_carry_dataset
    from trade_rl.integrations.market_store.carry_source import funding_coverage_gap

    spot, perp = series(perpetual=False), series(perpetual=True)
    base = np.asarray(perp.funding_event_count).copy()
    variants = {"full": base.copy(), "none": np.zeros_like(base)}
    late = base.copy()
    late[:20] = 0
    variants["late_start"] = late
    early = base.copy()
    early[-20:] = 0
    variants["early_end"] = early
    hole = base.copy()
    hole[15:40] = 0
    variants["hole"] = hole
    for name, counts in variants.items():
        changed = replace(
            perp,
            funding_event_count=counts,
            funding_available=counts > 0,
            funding_rate=counts * 0.0001,
        )
        try:
            assemble_carry_dataset({"BTCUSDT": (spot, changed)})
            accepted = True
        except ValueError:
            accepted = False
        assert (funding_coverage_gap(counts) is None) is accepted, name


def test_coverage_evidence_is_optional_and_copied() -> None:
    assert CarryCoverageError("x").evidence is None
    evidence = {"kind": "gap"}
    error = CarryCoverageError("x", evidence)
    evidence["kind"] = "changed"
    assert error.evidence == {"kind": "gap"}
