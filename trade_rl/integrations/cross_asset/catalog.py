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
    """One daily series, its point-in-time rules and its role in the features.

    ``revision_risk`` is ``low`` only when a measurement supports it (see
    ``revision_evidence``); ``possible`` means revisions were measured and
    ``unknown`` means no vintage history could be compared. Only ``low`` series
    are default composite members; every other series is opt-in.
    """

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
    revision_risk: str = "unknown"
    revision_evidence: str = "no vintage history available to compare"

    def __post_init__(self) -> None:
        if self.revision_risk not in REVISION_RISKS:
            raise ValueError(f"revision_risk must be one of {sorted(REVISION_RISKS)}")
        if not self.revision_evidence:
            raise ValueError("revision_evidence must not be empty")

    @property
    def default_member(self) -> bool:
        return self.revision_risk == "low"

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
            "revision_evidence": self.revision_evidence,
        }


REVISION_RISKS = frozenset({"low", "possible", "unknown"})

_NY = "America/New_York"
_DAILY = ReleaseRule(lag_business_days=1)
_WEEKLY_MONDAY = ReleaseRule(lag_business_days=1, weekly_release_weekday=0)
_DAILY_STALE_HOURS = 10 * 24.0
_WEEKLY_STALE_HOURS = 16 * 24.0

# ALFRED measurement, 2026-11-10: the vintage published on VINTAGE_DATE compared
# with the current values over the dates both contain. Reproduce with
# https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=<ID>&vintage_date=<date>
_ALFRED = "ALFRED vintage 2023-11-15 vs current"
_NO_VINTAGE_YAHOO = (
    "no vintage history exists for this endpoint; revisions can only be seen by "
    "repeated fetches (status counts them)"
)


def _fred(
    key: str,
    kind: SeriesKind,
    group: str,
    description: str,
    *,
    risk: str,
    evidence: str,
    close: time = time(17, 0),
    tz: str = _NY,
    release: ReleaseRule = _DAILY,
    stale: float = _DAILY_STALE_HOURS,
    breadth: bool = False,
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
        revision_risk=risk,
        revision_evidence=evidence,
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
        revision_risk="unknown",
        revision_evidence=_NO_VINTAGE_YAHOO,
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
        revision_risk="unknown",
        revision_evidence="source unavailable (bot check); never measured",
    )


P, L = SeriesKind.PRICE, SeriesKind.LEVEL

# Continuous front-month futures (GC=F, SI=F, HG=F, CL=F, BZ=F) are deliberately
# absent: their series jump on contract rolls and the only repair, back-adjusting
# the history, would use future knowledge. No key-less spot gold, silver or
# copper series exists (Yahoo XAUUSD=X, XAGUSD=X and XCUUSD=X answer 404).
CATALOG: tuple[SeriesSpec, ...] = (
    # --- US rates (H.15 / Treasury): percent levels, measured in differences.
    _fred(
        "DGS2",
        L,
        "rates",
        "US Treasury 2y constant maturity yield, %",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 11862 values changed",
    ),
    _fred(
        "DGS10",
        L,
        "rates",
        "US Treasury 10y constant maturity yield, %",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 15454 changed (1 of 14484 against the 2020-01-02 vintage)",
    ),
    _fred(
        "T10Y2Y",
        L,
        "rates",
        "10y minus 2y Treasury spread, pct points",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 11863 values changed",
    ),
    _fred(
        "DFII10",
        L,
        "rates",
        "US 10y TIPS real yield, %",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 5223 values changed",
    ),
    # --- Dollar and FX (H.10 noon buying rates, released weekly on Mondays).
    _fred(
        "DTWEXBGS",
        P,
        "dollar",
        "Nominal broad US dollar index",
        risk="possible",
        evidence=f"{_ALFRED}: 1707 of 4479 values changed (max 0.5%, 2018-2023)",
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
        risk="low",
        evidence=f"{_ALFRED}: 0 of 13251 changed (0 of 12284 against 2020-01-02)",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
    ),
    _fred(
        "DEXUSEU",
        P,
        "fx",
        "USD per EUR",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 6236 values changed",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
    ),
    _fred(
        "DEXUSUK",
        P,
        "fx",
        "USD per GBP",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 13257 values changed",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
    ),
    _fred(
        "DEXCHUS",
        P,
        "fx",
        "CNY per USD",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 10697 values changed",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
    ),
    _fred(
        "DEXUSAL",
        P,
        "fx",
        "USD per AUD",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 13250 values changed",
        close=time(12, 0),
        release=_WEEKLY_MONDAY,
        stale=_WEEKLY_STALE_HOURS,
    ),
    # --- Risk appetite.
    _fred(
        "VIXCLS",
        P,
        "risk",
        "CBOE VIX close",
        risk="low",
        evidence=f"{_ALFRED}: 8 of 8544 values changed (max 4.2%, last in 2019)",
        close=time(16, 15),
    ),
    _fred(
        "BAMLH0A0HYM2",
        L,
        "risk",
        "ICE BofA US high yield option-adjusted spread, %",
        risk="possible",
        evidence=(
            "ICE licensing keeps a rolling three year window: the 2023-11-15 "
            "vintage held 4 observations, so history is rewritten even though "
            "vintages 2024-06-03 and 2025-06-02 show 0 changed"
        ),
    ),
    _fred(
        "BAMLC0A0CM",
        L,
        "risk",
        "ICE BofA US corporate option-adjusted spread, %",
        risk="possible",
        evidence=(
            "ICE licensing keeps a rolling three year window (same series "
            "family as BAMLH0A0HYM2); the 2025-06-02 vintage shows 0 changed"
        ),
    ),
    # --- Equity indices published on FRED. SP500 was observed three sessions
    # behind at the first fetch, so it carries a three business day lag.
    _fred(
        "SP500",
        P,
        "equity",
        "S&P 500 (FRED keeps about ten years)",
        risk="unknown",
        evidence="ALFRED returns 404 for this series: no vintage to compare",
        close=time(16, 0),
        release=ReleaseRule(lag_business_days=3),
        breadth=True,
    ),
    _fred(
        "NASDAQCOM",
        P,
        "equity",
        "NASDAQ composite",
        risk="possible",
        evidence=f"{_ALFRED}: 996 of 13275 values changed (max 1.4%, mostly 2018-2020)",
        close=time(16, 0),
        breadth=True,
    ),
    _fred(
        "NIKKEI225",
        P,
        "equity",
        "Nikkei 225",
        risk="low",
        evidence=f"{_ALFRED}: 0 of 18510 values changed",
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
        risk="possible",
        evidence=f"{_ALFRED}: 0 changed but 743 of 9539 earlier values are now missing",
        release=ReleaseRule(lag_business_days=6),
        stale=15 * 24.0,
        breadth=True,
    ),
    _fred(
        "DCOILBRENTEU",
        P,
        "commodity",
        "Brent crude spot, USD per barrel",
        risk="possible",
        evidence=f"{_ALFRED}: 0 changed but 890 of 9260 earlier values are now missing",
        release=ReleaseRule(lag_business_days=6),
        stale=15 * 24.0,
        breadth=True,
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
    "REVISION_RISKS",
    "SOURCES",
    "SeriesKind",
    "SeriesSpec",
    "SourceSpec",
    "series_by_key",
]
