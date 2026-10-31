"""The embargo decision comes from the URL, enforced again at the fetcher itself."""

from __future__ import annotations

import urllib.error
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.integrations.market_store.support import (
    FakeFetcher,
    MemoryStore,
    archive_bytes,
)
from trade_rl.integrations.market_store.archive_url import (
    ArchiveSpecMismatchError,
    UnrecognizedArchiveUrlError,
    parse_archive_url,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    Fs1WarmupSpanError,
    SealedWindowEmbargoError,
)
from trade_rl.integrations.market_store.fetch import (
    RedirectRefused,
    VisionFetcher,
    _NoRedirect,
)
from trade_rl.integrations.market_store.ingest import run_ingestion
from trade_rl.integrations.market_store.plan import IngestionPlan

URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-88"
ROOT = "https://data.binance.vision/data"
SPOT = f"{ROOT}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h"
UM = f"{ROOT}/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h"
UNIVERSE = Universe(symbols=("BTCUSDT",), markets=(StoreMarket.SPOT,))
POLICIES = {
    "enforced": EmbargoPolicy.enforced(),
    "warmup": EmbargoPolicy.with_warmup(URL),
}


def day(year: int, month: int, number: int = 1) -> datetime:
    return datetime(year, month, number, tzinfo=UTC)


# -- parsing ---------------------------------------------------------------------


def test_parser_reads_market_symbol_interval_and_period_from_the_url() -> None:
    spot = parse_archive_url(f"{SPOT}-2026-07.zip")
    assert (spot.market, spot.symbol, spot.interval) == (
        StoreMarket.SPOT,
        "BTCUSDT",
        "1h",
    )
    assert (spot.period_kind, spot.period_start, spot.period_end) == (
        PeriodKind.MONTHLY,
        day(2026, 7),
        day(2026, 8),
    )
    assert not spot.is_checksum
    december = parse_archive_url(f"{UM}-2026-12.zip.CHECKSUM")
    assert december.market is StoreMarket.USDS_M and december.is_checksum
    assert december.period_end == day(2027, 1)
    daily = parse_archive_url(
        f"{ROOT}/spot/daily/klines/ETHUSDT/4h/ETHUSDT-4h-2026-10-31.zip"
    )
    assert (daily.period_kind, daily.period_end) == (PeriodKind.DAILY, day(2026, 11))
    funding = parse_archive_url(
        f"{ROOT}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-08.zip"
    )
    assert funding.kind is ArchiveKind.FUNDING_RATE and funding.interval is None
    assert funding.period_start == day(2026, 8)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://example.com/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip",
        "http://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip",
        f"{SPOT}-2026-07.zip?redirect=https://evil.example/",
        f"{SPOT}-2026-07.zip/../../x",
        f"{SPOT}-2026-07.zip.CHECKSUM.zip",
        f"{SPOT}-2026-13.zip",
        f"{SPOT}-2026-07-15.zip",  # monthly directory with a daily name
        f"{ROOT}/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip",  # reverse
        f"{ROOT}/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-02-30.zip",
        f"{ROOT}/spot/monthly/klines/BTCUSDT/1h/ETHUSDT-1h-2026-07.zip",
        f"{ROOT}/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-4h-2026-07.zip",
        f"{ROOT}/futures/cm/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip",
        f"{ROOT}/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2026-07.zip",
        f"{ROOT}/spot/monthly/klines/btcusdt/1h/btcusdt-1h-2026-07.zip",
        f"{ROOT}/futures/um/daily/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-07.zip",
    ],
)
def test_unrecognised_urls_are_refused(url: str) -> None:
    with pytest.raises(UnrecognizedArchiveUrlError):
        parse_archive_url(url)


# -- a declared spec that disagrees with its URL ---------------------------------


def spec_for(
    url: str,
    *,
    kind: ArchiveKind = ArchiveKind.KLINES,
    market: StoreMarket = StoreMarket.SPOT,
    symbol: str = "BTCUSDT",
    interval: str | None = "1h",
    period_kind: PeriodKind = PeriodKind.MONTHLY,
    start: datetime = day(2026, 6),
    end: datetime = day(2026, 7),
) -> ArchiveSpec:
    return ArchiveSpec(
        kind=kind,
        market=market,
        symbol=symbol,
        interval=interval,
        period_kind=period_kind,
        period_start=start,
        period_end=end,
        url=url,
        checksum_url=f"{url}.CHECKSUM",
    )


def run(spec: ArchiveSpec, policy: EmbargoPolicy) -> tuple[FakeFetcher, MemoryStore]:
    fetcher, store = FakeFetcher(), MemoryStore()
    for url in (spec.url, spec.checksum_url):
        fetcher.files[url] = b"would be downloaded"
    plan = IngestionPlan(
        UNIVERSE, spec.period_start, spec.period_end, policy, (spec,), ()
    )
    with pytest.raises((ArchiveSpecMismatchError, SealedWindowEmbargoError)):
        run_ingestion(store, fetcher, plan, code_commit="t", command_line="t")
    return fetcher, store


@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize("month", ["2026-11", "2026-08", "2026-12", "2026-10"])
def test_a_false_declared_period_never_reaches_the_fetcher(
    name: str, month: str
) -> None:
    policy = POLICIES[name]
    spec = spec_for(f"{SPOT}-{month}.zip")  # URL says `month`, declaration says 2026-06
    fetcher, store = run(spec, policy)
    assert fetcher.requested == []
    assert store.runs == {} and store.sources == {} and store.klines == {}


@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": "ETHUSDT"},
        {"interval": "4h"},
        {"market": StoreMarket.USDS_M},
        {"period_kind": PeriodKind.DAILY, "end": day(2026, 6, 2)},
        {"kind": ArchiveKind.FUNDING_RATE, "interval": None},
        {"start": day(2026, 7), "end": day(2026, 8)},
    ],
    ids=["symbol", "interval", "market", "period-kind", "kind", "period"],
)
def test_any_declared_field_disagreeing_with_the_url_is_refused(
    name: str, overrides: dict[str, Any]
) -> None:
    spec = spec_for(f"{SPOT}-2026-06.zip", **overrides)
    fetcher, store = run(spec, POLICIES[name])
    assert fetcher.requested == [] and store.runs == {}


@pytest.mark.parametrize("name", list(POLICIES))
def test_a_truthful_spec_for_a_sealed_url_is_refused_from_the_url_period(
    name: str,
) -> None:
    spec = spec_for(f"{SPOT}-2026-11.zip", start=day(2026, 11), end=day(2026, 12))
    fetcher, store = run(spec, POLICIES[name])
    assert fetcher.requested == [] and store.runs == {}


def test_a_truthful_warmup_spec_needs_the_authorization() -> None:
    spec = spec_for(f"{SPOT}-2026-08.zip", start=day(2026, 8), end=day(2026, 9))
    fetcher, _ = run(spec, POLICIES["enforced"])
    assert fetcher.requested == []
    fetcher = FakeFetcher()
    fetcher.add(spec.url, archive_bytes(spec))
    store = MemoryStore()
    plan = IngestionPlan(
        UNIVERSE, spec.period_start, spec.period_end, POLICIES["warmup"], (spec,), ()
    )
    run_ingestion(store, fetcher, plan, code_commit="t", command_line="t")
    assert fetcher.requested == [spec.url, spec.checksum_url]


# -- the fetcher refuses on its own, before any request -----------------------------


class Response:
    def __init__(self, url: str, status: int = 200, body: bytes = b"zip") -> None:
        self._url, self.status, self._body = url, status, body

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, limit: int = -1) -> bytes:
        return self._body


class Opener:
    def __init__(self, respond: Any = None) -> None:
        self.calls: list[str] = []
        self._respond = respond

    def __call__(self, request: Any, timeout: float = 0.0) -> Response:
        self.calls.append(request.full_url)
        if self._respond is not None:
            return self._respond(request.full_url)
        return Response(request.full_url)


def fetcher_with(opener: Opener, *, authorized: bool = False) -> VisionFetcher:
    return VisionFetcher(
        warmup_authorization_url=URL if authorized else None,
        retry_backoff_seconds=0.0,
        _opener=opener,
    )


@pytest.mark.parametrize("suffix", ["", ".CHECKSUM"])
@pytest.mark.parametrize("month", ["2026-08", "2026-09", "2026-10"])
def test_fetcher_refuses_warmup_urls_without_authorization(
    month: str, suffix: str
) -> None:
    opener = Opener()
    with pytest.raises(Fs1WarmupSpanError):
        fetcher_with(opener).fetch(f"{SPOT}-{month}.zip{suffix}")
    assert opener.calls == []


@pytest.mark.parametrize("authorized", [False, True])
@pytest.mark.parametrize("suffix", ["", ".CHECKSUM"])
@pytest.mark.parametrize("month", ["2026-11", "2026-12", "2027-03"])
def test_fetcher_never_takes_a_sealed_url(
    authorized: bool, month: str, suffix: str
) -> None:
    opener = Opener()
    with pytest.raises(SealedWindowEmbargoError):
        fetcher_with(opener, authorized=authorized).fetch(f"{UM}-{month}.zip{suffix}")
    with pytest.raises(SealedWindowEmbargoError):
        fetcher_with(opener, authorized=authorized).fetch(
            f"{ROOT}/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{month}-05.zip"
        )
    assert opener.calls == []


@pytest.mark.parametrize("authorized", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "",
        "garbage",
        "https://evil.example/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-07.zip",
        f"{SPOT}-2026-07.zip?x=1",
        f"{SPOT}-2026-07-15.zip",
        "https://data.binance.vision/data/spot/monthly/klines/../../../etc/passwd",
    ],
)
def test_fetcher_refuses_unparseable_urls(authorized: bool, url: str) -> None:
    opener = Opener()
    with pytest.raises(UnrecognizedArchiveUrlError):
        fetcher_with(opener, authorized=authorized).fetch(url)
    assert opener.calls == []


def test_fetcher_takes_pre_warmup_urls_and_authorized_warmup_urls_only() -> None:
    opener = Opener()
    assert fetcher_with(opener).fetch(f"{SPOT}-2026-07.zip").status == 200
    assert fetcher_with(opener).fetch(f"{UM}-2021-01.zip.CHECKSUM").status == 200
    assert (
        fetcher_with(opener, authorized=True).fetch(f"{SPOT}-2026-10.zip").status == 200
    )
    assert len(opener.calls) == 3


# -- redirects -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "https://evil.example/anything.zip",
        f"{SPOT}-2026-11.zip",
        f"{SPOT}-2026-08.zip",
        "http://data.binance.vision/data/x",
    ],
)
def test_redirects_are_never_followed(target: str) -> None:
    handler = _NoRedirect()
    request = urllib_request(f"{SPOT}-2026-07.zip")
    assert handler.redirect_request(request, None, 302, "Found", {}, target) is None

    def redirect(url: str) -> Response:
        raise urllib.error.HTTPError(url, 302, "Found", {"Location": target}, None)  # type: ignore[arg-type]

    opener = Opener(redirect)
    result = fetcher_with(opener).fetch(f"{SPOT}-2026-07.zip")
    assert result.status == 302 and result.body is None
    assert opener.calls == [f"{SPOT}-2026-07.zip"]  # the target was never requested


@pytest.mark.parametrize(
    "final, error",
    [
        ("https://evil.example/anything.zip", UnrecognizedArchiveUrlError),
        (f"{SPOT}-2026-11.zip", SealedWindowEmbargoError),
        (f"{SPOT}-2026-08.zip", Fs1WarmupSpanError),
        (f"{SPOT}-2021-01.zip", RedirectRefused),
    ],
)
def test_a_response_from_a_different_final_url_is_refused(
    final: str, error: type[Exception]
) -> None:
    opener = Opener(lambda url: Response(final))
    with pytest.raises(error):
        fetcher_with(opener).fetch(f"{SPOT}-2026-07.zip")
    assert opener.calls == [f"{SPOT}-2026-07.zip"]


def urllib_request(url: str) -> Any:
    import urllib.request

    return urllib.request.Request(url)
