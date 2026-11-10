"""Series and source catalog for the cross-asset market context.

Every source is key-less and public. A source that cannot be read is recorded in
the manifest and skipped; nothing here bypasses a login, key or bot check.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from datetime import time
from enum import StrEnum

from trade_rl.integrations.cross_asset.pit import ReleaseRule


class SeriesKind(StrEnum):
    """How changes are measured: log returns for prices, differences for levels."""

    PRICE = "price"
    LEVEL = "level"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    url_template: str
    terms_url: str
    terms_note: str


SOURCES: dict[str, SourceSpec] = {
    "fred": SourceSpec(
        name="fred",
        url_template="https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}",
        terms_url="https://fred.stlouisfed.org/legal/",
        terms_note=(
            "FRED permits personal, non-commercial and educational use with "
            "attribution; series with a third-party copyright notice (S&P 500, "
            "NASDAQ, Nikkei, ICE BofA) are for personal use only. The CSV graph "
            "endpoint needs no key. Raw data stays in the ignored data directory."
        ),
    ),
    "yahoo": SourceSpec(
        name="yahoo",
        url_template=(
            "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            "?period1=631152000&period2=4102444800&interval=1d&events=history"
        ),
        terms_url="https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
        terms_note=(
            "Unofficial, undocumented chart endpoint (the one yfinance uses); no "
            "key. The explicit 1990..2100 period is required: range=max silently "
            "degrades to monthly bars. No licence for redistribution or "
            "commercial use was obtained and the linked terms were not "
            "reviewed by the implementer: treat as best-effort research input, "
            "confirm the terms before relying on it, never commit the data."
        ),
    ),
    "stooq": SourceSpec(
        name="stooq",
        url_template="https://stooq.com/q/d/l/?s={symbol}&i=d",
        terms_url="https://stooq.com/",
        terms_note=(
            "Daily CSV download endpoint; terms not reviewed by the implementer. "
            "On 2026-11-10 it answered with a "
            "JavaScript proof-of-work bot check instead of CSV. That is bot "
            "detection and is not bypassed: the source is recorded as "
            "unavailable and skipped until it serves CSV to a plain client."
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One daily series, its point-in-time rules and its role in the features."""

    key: str
    source: str
    symbol: str
    kind: SeriesKind
    group: str
    description: str
    venue_tz: str
    close_local: time
    release: ReleaseRule
    max_staleness_hours: float
    trend_breadth: bool = False
    revision_risk: str = "low"

    def url(self) -> str:
        template = SOURCES[self.source].url_template
        return template.format(symbol=urllib.parse.quote(self.symbol, safe=""))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": self.source,
            "symbol": self.symbol,
            "kind": self.kind.value,
            "group": self.group,
            "venue_tz": self.venue_tz,
            "close_local": self.close_local.isoformat(),
            "release": self.release.canonical_payload(),
            "max_staleness_hours": self.max_staleness_hours,
            "trend_breadth": self.trend_breadth,
            "revision_risk": self.revision_risk,
        }


_NY = "America/New_York"
_DAILY = ReleaseRule(lag_business_days=1)
_WEEKLY_MONDAY = ReleaseRule(lag_business_days=1, weekly_release_weekday=0)
_DAILY_STALE_HOURS = 10 * 24.0
_WEEKLY_STALE_HOURS = 16 * 24.0


def _fred(
    key: str,
    kind: SeriesKind,
    group: str,
    description: str,
    *,
    close: time = time(17, 0),
    tz: str = _NY,
    release: ReleaseRule = _DAILY,
    stale: float = _DAILY_STALE_HOURS,
    breadth: bool = False,
    revision_risk: str = "low",
) -> SeriesSpec:
    return SeriesSpec(
        key=key,
        source="fred",
        symbol=key,
        kind=kind,
        group=group,
        description=description,
        venue_tz=tz,
        close_local=close,
        release=release,
        max_staleness_hours=stale,
        trend_breadth=breadth,
        revision_risk=revision_risk,
    )


def _yahoo(
    key: str,
    symbol: str,
    group: str,
    description: str,
    *,
    tz: str,
    close: time,
    breadth: bool = True,
) -> SeriesSpec:
    return SeriesSpec(
        key=key,
        source="yahoo",
        symbol=symbol,
        kind=SeriesKind.PRICE,
        group=group,
        description=description,
        venue_tz=tz,
        close_local=close,
        release=_DAILY,
        max_staleness_hours=_DAILY_STALE_HOURS,
        trend_breadth=breadth,
        revision_risk="low",
    )


def _stooq(
    key: str, symbol: str, group: str, description: str, *, tz: str, close: time
) -> SeriesSpec:
    return SeriesSpec(
        key=key,
        source="stooq",
        symbol=symbol,
        kind=SeriesKind.PRICE,
        group=group,
        description=description,
        venue_tz=tz,
        close_local=close,
        release=_DAILY,
        max_staleness_hours=_DAILY_STALE_HOURS,
        trend_breadth=False,
        revision_risk="low",
    )


P, L = SeriesKind.PRICE, SeriesKind.LEVEL

CATALOG: tuple[SeriesSpec, ...] = (
    # --- US rates (H.15 / Treasury): percent levels, measured in differences.
    _fred("DGS2", L, "rates", "US Treasury 2y constant maturity yield, %"),
    _fred("DGS10", L, "rates", "US Treasury 10y constant maturity yield, %"),
    _fred("T10Y2Y", L, "rates", "10y minus 2y Treasury spread, pct points"),
    _fred("DFII10", L, "rates", "US 10y TIPS real yield, %"),
    # --- Dollar and FX (H.10 noon buying rates, released weekly on Mondays).
    _fred(
        "DTWEXBGS",
        P,
        "dollar",
        "Nominal broad US dollar index",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    _fred(
        "DEXJPUS",
        P,
        "fx",
        "JPY per USD",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    _fred(
        "DEXUSEU",
        P,
        "fx",
        "USD per EUR",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    _fred(
        "DEXUSUK",
        P,
        "fx",
        "USD per GBP",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    _fred(
        "DEXCHUS",
        P,
        "fx",
        "CNY per USD",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    _fred(
        "DEXUSAL",
        P,
        "fx",
        "USD per AUD",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
        breadth=True,
    ),
    # --- Risk appetite.
    _fred("VIXCLS", P, "risk", "CBOE VIX close", close=time(16, 15)),
    _fred(
        "BAMLH0A0HYM2",
        L,
        "risk",
        "ICE BofA US high yield option-adjusted spread, %",
        revision_risk="possible",
    ),
    _fred(
        "BAMLC0A0CM",
        L,
        "risk",
        "ICE BofA US corporate option-adjusted spread, %",
        revision_risk="possible",
    ),
    # --- Equity indices published on FRED. SP500 was observed three sessions
    # behind at the first fetch, so it carries a three business day lag.
    _fred(
        "SP500",
        P,
        "equity",
        "S&P 500 (FRED keeps about ten years)",
        close=time(16, 0),
        release=ReleaseRule(lag_business_days=3),
        breadth=True,
    ),
    _fred(
        "NASDAQCOM",
        P,
        "equity",
        "NASDAQ composite",
        close=time(16, 0),
        breadth=True,
    ),
    _fred(
        "NIKKEI225",
        P,
        "equity",
        "Nikkei 225",
        close=time(15, 30),
        tz="Asia/Tokyo",
        breadth=True,
    ),
    # --- Energy spot (EIA). Observed about one week behind, hence a long lag.
    _fred(
        "DCOILWTICO",
        P,
        "commodity",
        "WTI crude spot, USD per barrel",
        release=ReleaseRule(lag_business_days=6),
        stale=15 * 24.0,
    ),
    _fred(
        "DCOILBRENTEU",
        P,
        "commodity",
        "Brent crude spot, USD per barrel",
        release=ReleaseRule(lag_business_days=6),
        stale=15 * 24.0,
    ),
    # --- Yahoo extras that FRED does not carry (no overlap with FRED above).
    _yahoo(
        "y_gdaxi", "^GDAXI", "equity", "DAX", tz="Europe/Berlin", close=time(17, 30)
    ),
    _yahoo(
        "y_ftse", "^FTSE", "equity", "FTSE 100", tz="Europe/London", close=time(16, 30)
    ),
    _yahoo(
        "y_hsi", "^HSI", "equity", "Hang Seng", tz="Asia/Hong_Kong", close=time(16, 0)
    ),
    _yahoo(
        "y_ssec",
        "000001.SS",
        "equity",
        "Shanghai composite",
        tz="Asia/Shanghai",
        close=time(15, 0),
    ),
    _yahoo(
        "y_dxy", "DX-Y.NYB", "dollar", "ICE US dollar index", tz=_NY, close=time(17, 0)
    ),
    _yahoo(
        "y_gold",
        "GC=F",
        "commodity",
        "COMEX gold front future",
        tz=_NY,
        close=time(17, 0),
    ),
    _yahoo(
        "y_silver",
        "SI=F",
        "commodity",
        "COMEX silver front future",
        tz=_NY,
        close=time(17, 0),
    ),
    _yahoo(
        "y_copper",
        "HG=F",
        "commodity",
        "COMEX copper front future",
        tz=_NY,
        close=time(17, 0),
    ),
    _yahoo(
        "y_wti",
        "CL=F",
        "commodity",
        "NYMEX WTI front future",
        tz=_NY,
        close=time(17, 0),
    ),
    _yahoo(
        "y_brent",
        "BZ=F",
        "commodity",
        "ICE Brent front future",
        tz=_NY,
        close=time(17, 0),
    ),
    # --- Stooq (currently behind a bot check; probed once per fetch run).
    _stooq("stq_spx", "^spx", "equity", "S&P 500", tz=_NY, close=time(16, 0)),
    _stooq("stq_ndq", "^ndq", "equity", "NASDAQ 100", tz=_NY, close=time(16, 0)),
    _stooq(
        "stq_nkx", "^nkx", "equity", "Nikkei 225", tz="Asia/Tokyo", close=time(15, 30)
    ),
    _stooq("stq_dax", "^dax", "equity", "DAX", tz="Europe/Berlin", close=time(17, 30)),
    _stooq(
        "stq_ftm", "^ftm", "equity", "FTSE 100", tz="Europe/London", close=time(16, 30)
    ),
    _stooq(
        "stq_hsi", "^hsi", "equity", "Hang Seng", tz="Asia/Hong_Kong", close=time(16, 0)
    ),
    _stooq(
        "stq_shc",
        "^shc",
        "equity",
        "Shanghai composite",
        tz="Asia/Shanghai",
        close=time(15, 0),
    ),
    _stooq("stq_eurusd", "eurusd", "fx", "EUR/USD", tz=_NY, close=time(17, 0)),
    _stooq("stq_usdjpy", "usdjpy", "fx", "USD/JPY", tz=_NY, close=time(17, 0)),
    _stooq("stq_gbpusd", "gbpusd", "fx", "GBP/USD", tz=_NY, close=time(17, 0)),
    _stooq("stq_audusd", "audusd", "fx", "AUD/USD", tz=_NY, close=time(17, 0)),
    _stooq("stq_usdcnh", "usdcnh", "fx", "USD/CNH", tz=_NY, close=time(17, 0)),
    _stooq("stq_xauusd", "xauusd", "commodity", "Gold spot", tz=_NY, close=time(17, 0)),
    _stooq(
        "stq_xagusd", "xagusd", "commodity", "Silver spot", tz=_NY, close=time(17, 0)
    ),
    _stooq("stq_cl_f", "cl.f", "commodity", "WTI future", tz=_NY, close=time(17, 0)),
    _stooq("stq_hg_f", "hg.f", "commodity", "Copper future", tz=_NY, close=time(17, 0)),
)

_BY_KEY = {spec.key: spec for spec in CATALOG}
if len(_BY_KEY) != len(CATALOG):
    raise RuntimeError("cross-asset catalog keys must be unique")


def series_by_key(key: str) -> SeriesSpec:
    try:
        return _BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"unknown cross-asset series: {key}") from error


__all__ = [
    "CATALOG",
    "SOURCES",
    "SeriesKind",
    "SeriesSpec",
    "SourceSpec",
    "series_by_key",
]
