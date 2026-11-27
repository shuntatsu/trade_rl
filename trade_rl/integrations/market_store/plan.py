"""Pure, network-free ingestion planning with the FS1 embargo applied.

The URL arithmetic is the existing Binance Vision planner
(:func:`~trade_rl.integrations.binance.plan_vision_kline_urls` and
:func:`~trade_rl.integrations.binance.vision_funding_url`); this module only adds the
store policy on top:

* monthly archives are used for completed months that have had time to publish
  (``monthly_publication_lag``); the current partial month is covered by daily
  archives up to (excluding) today;
* funding rates exist only as monthly Vision archives, so a month that is not yet
  published or that would reach the embargo boundary is *skipped and reported*,
  never fetched;
* an archive whose period reaches the embargo boundary is never planned, including a
  monthly archive straddling the boundary (its in-bounds days are planned daily);
* the opt-in kinds ``metrics`` and ``bookDepth`` exist only as USD-M **daily** archives:
  one per symbol and UTC day in ``[start, today)``, never for spot or monthly periods,
  and (like every archive) never past the embargo boundary. The default ``kinds`` stay
  klines and funding rates, so a plan without the opt-in is byte-for-byte unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trade_rl.integrations.binance import (
    plan_vision_kline_urls,
    vision_funding_url,
    vision_kline_url,
)
from trade_rl.integrations.market_store.archive_url import (
    vision_daily_derivatives_url,
)
from trade_rl.integrations.market_store.contracts import (
    DAILY_USDS_M_ONLY_KINDS,
    DEFAULT_ARCHIVE_KINDS,
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
    Universe,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy

DEFAULT_MONTHLY_PUBLICATION_LAG = timedelta(days=7)

_PERIOD = re.compile(r"-(\d{4})-(\d{2})(?:-(\d{2}))?\.zip$")


@dataclass(frozen=True, slots=True)
class PlanSkip:
    """A period that is deliberately not planned, with the reason."""

    kind: ArchiveKind
    market: StoreMarket
    symbol: str
    period_start: datetime
    period_end: datetime
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "market": self.market.value,
            "period_end": self.period_end.isoformat(),
            "period_start": self.period_start.isoformat(),
            "reason": self.reason,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    universe: Universe
    start: datetime
    end: datetime
    embargo: EmbargoPolicy
    specs: tuple[ArchiveSpec, ...]
    skipped: tuple[PlanSkip, ...]
    kinds: tuple[ArchiveKind, ...] = DEFAULT_ARCHIVE_KINDS

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "archives": [spec.to_dict() for spec in self.specs],
            "archives_planned": len(self.specs),
            "closed_from": self.embargo.start.isoformat(),
            "fs1_warmup_allowed": self.embargo.warmup_allowed,
            "fs1_warmup_authorization_url": self.embargo.warmup_authorization_url,
            "fs1_warmup_start": self.embargo.warmup_start.isoformat(),
            "sealed_window_start": self.embargo.sealed_start.isoformat(),
            "end": self.end.isoformat(),
            "skipped": [item.to_dict() for item in self.skipped],
            "start": self.start.isoformat(),
            "universe": self.universe.to_dict(),
        }
        if self.kinds != DEFAULT_ARCHIVE_KINDS:
            # recorded only for an explicit opt-in, so default plans/runs keep the
            # exact request document they always had
            payload["kinds"] = [kind.value for kind in self.kinds]
        return payload


def _utc_day(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _floor_day(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start(value: datetime) -> datetime:
    return _floor_day(value).replace(day=1)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def _period_from_url(url: str) -> tuple[PeriodKind, datetime, datetime]:
    match = _PERIOD.search(url)
    if match is None:
        raise ValueError(f"cannot read the archive period from {url}")
    year, month, day = match.group(1), match.group(2), match.group(3)
    if day is None:
        start = datetime(int(year), int(month), 1, tzinfo=UTC)
        return PeriodKind.MONTHLY, start, _next_month(start)
    start = datetime(int(year), int(month), int(day), tzinfo=UTC)
    return PeriodKind.DAILY, start, start + timedelta(days=1)


def _kline_specs(
    market: StoreMarket,
    symbol: str,
    interval: str,
    *,
    start: datetime,
    end: datetime,
    monthly_end: datetime,
) -> list[ArchiveSpec]:
    urls: list[str] = []
    head_end = min(end, monthly_end)
    if start < head_end:
        urls.extend(
            plan_vision_kline_urls(market.binance, symbol, interval, start, head_end)
        )
    day = max(start, monthly_end)
    while day < end:
        urls.append(vision_kline_url(market.binance, symbol, interval, day))
        day += timedelta(days=1)
    specs: list[ArchiveSpec] = []
    for url in dict.fromkeys(urls):
        period_kind, period_start, period_end = _period_from_url(url)
        specs.append(
            ArchiveSpec(
                kind=ArchiveKind.KLINES,
                market=market,
                symbol=symbol,
                interval=interval,
                period_kind=period_kind,
                period_start=period_start,
                period_end=period_end,
                url=url,
                checksum_url=f"{url}.CHECKSUM",
            )
        )
    return specs


def _funding_specs(
    specs: list[ArchiveSpec],
    skipped: list[PlanSkip],
    market: StoreMarket,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    monthly_end: datetime,
) -> None:
    month = _month_start(start)
    while month < end:
        following = _next_month(month)
        if following <= end and following <= monthly_end:
            url = vision_funding_url(market.binance, symbol, month)
            specs.append(
                ArchiveSpec(
                    kind=ArchiveKind.FUNDING_RATE,
                    market=market,
                    symbol=symbol,
                    interval=None,
                    period_kind=PeriodKind.MONTHLY,
                    period_start=month,
                    period_end=following,
                    url=url,
                    checksum_url=f"{url}.CHECKSUM",
                )
            )
        else:
            reason = (
                "funding rates exist only as monthly archives and this "
                "month is not complete and published inside the range"
            )
            skipped.append(
                PlanSkip(
                    ArchiveKind.FUNDING_RATE,
                    market,
                    symbol,
                    month,
                    following,
                    reason,
                )
            )
        month = following


def _daily_derivatives_specs(
    kind: ArchiveKind,
    market: StoreMarket,
    symbol: str,
    *,
    start: datetime,
    end: datetime,
) -> list[ArchiveSpec]:
    specs: list[ArchiveSpec] = []
    day = start
    while day < end:
        url = vision_daily_derivatives_url(kind, market, symbol, day)
        specs.append(
            ArchiveSpec(
                kind=kind,
                market=market,
                symbol=symbol,
                interval=None,
                period_kind=PeriodKind.DAILY,
                period_start=day,
                period_end=day + timedelta(days=1),
                url=url,
                checksum_url=f"{url}.CHECKSUM",
            )
        )
        day += timedelta(days=1)
    return specs


def _checked_kinds(
    kinds: Sequence[ArchiveKind], universe: Universe
) -> tuple[ArchiveKind, ...]:
    resolved = tuple(ArchiveKind(kind) for kind in kinds)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("kinds must be a non-empty list without duplicates")
    opt_in = [kind for kind in resolved if kind in DAILY_USDS_M_ONLY_KINDS]
    if opt_in and StoreMarket.USDS_M not in universe.markets:
        names = ", ".join(kind.value for kind in opt_in)
        raise ValueError(
            f"{names} archives exist only for USD-M futures, but the universe has "
            "no usds_m market"
        )
    return resolved


def plan_ingestion(
    universe: Universe,
    *,
    start: datetime,
    end: datetime,
    now: datetime,
    embargo: EmbargoPolicy | None = None,
    monthly_publication_lag: timedelta = DEFAULT_MONTHLY_PUBLICATION_LAG,
    kinds: Sequence[ArchiveKind] = DEFAULT_ARCHIVE_KINDS,
) -> IngestionPlan:
    """Plan every archive for ``[start, end)``; never reads the network or a clock."""

    planned_kinds = _checked_kinds(kinds, universe)
    policy = EmbargoPolicy.enforced() if embargo is None else embargo
    start_utc = _utc_day(start, field="start")
    end_utc = _utc_day(end, field="end")
    now_utc = _utc_day(now, field="now")
    if start_utc != _floor_day(start_utc) or end_utc != _floor_day(end_utc):
        raise ValueError("start and end must be UTC midnight boundaries")
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")
    policy.check_range_end(end_utc, what="requested")
    today = _floor_day(now_utc)
    effective_end = min(end_utc, today)
    monthly_end = _month_start(today - monthly_publication_lag)

    specs: list[ArchiveSpec] = []
    skipped: list[PlanSkip] = []
    if effective_end > start_utc:
        for market in universe.markets:
            for symbol in universe.symbols:
                if ArchiveKind.KLINES in planned_kinds:
                    specs.extend(
                        _kline_specs(
                            market,
                            symbol,
                            universe.interval,
                            start=start_utc,
                            end=effective_end,
                            monthly_end=monthly_end,
                        )
                    )
                if market is not StoreMarket.USDS_M:
                    continue
                if ArchiveKind.FUNDING_RATE in planned_kinds:
                    _funding_specs(
                        specs,
                        skipped,
                        market,
                        symbol,
                        start=start_utc,
                        end=effective_end,
                        monthly_end=monthly_end,
                    )
                for kind in planned_kinds:
                    if kind in DAILY_USDS_M_ONLY_KINDS:
                        specs.extend(
                            _daily_derivatives_specs(
                                kind,
                                market,
                                symbol,
                                start=start_utc,
                                end=effective_end,
                            )
                        )
    for spec in specs:
        policy.check_archive_end(spec.period_end, url=spec.url)
    return IngestionPlan(
        universe=universe,
        start=start_utc,
        end=effective_end if effective_end > start_utc else start_utc,
        embargo=policy,
        specs=tuple(specs),
        skipped=tuple(skipped),
        kinds=planned_kinds,
    )


__all__ = [
    "DEFAULT_MONTHLY_PUBLICATION_LAG",
    "IngestionPlan",
    "PlanSkip",
    "plan_ingestion",
]
