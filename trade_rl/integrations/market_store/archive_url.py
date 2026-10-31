"""Strict parsing of a Binance Vision archive URL into its market, symbol and period.

Every FS1 boundary decision about an archive is made from the period parsed out of the
URL itself, never from caller-declared metadata: a hand-built
:class:`~trade_rl.integrations.market_store.contracts.ArchiveSpec` that pairs a 2026-11
URL with a false 2026-06 period is refused here before any request is made.

Recognised forms (an optional ``.CHECKSUM`` suffix is allowed)::

    https://data.binance.vision/data/spot/{monthly|daily}/klines/SYM/INTERVAL/SYM-INTERVAL-PERIOD.zip
    https://data.binance.vision/data/futures/um/{monthly|daily}/klines/SYM/INTERVAL/SYM-INTERVAL-PERIOD.zip
    https://data.binance.vision/data/futures/um/monthly/fundingRate/SYM/SYM-fundingRate-YYYY-MM.zip

with ``PERIOD`` = ``YYYY-MM`` for ``monthly`` and ``YYYY-MM-DD`` for ``daily``.
Anything else is unrecognised and must not be fetched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    PeriodKind,
    StoreMarket,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy

VISION_PREFIX: Final = "https://data.binance.vision/data/"

_SYMBOL: Final = r"[A-Z0-9]{3,32}"
_KLINES: Final = re.compile(
    rf"^{re.escape(VISION_PREFIX)}(?P<root>spot|futures/um)/(?P<cadence>monthly|daily)"
    rf"/klines/(?P<symbol>{_SYMBOL})/(?P<interval>[0-9]+[mhd])"
    rf"/(?P<file_symbol>{_SYMBOL})-(?P<file_interval>[0-9]+[mhd])"
    r"-(?P<period>[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)\.zip(?P<checksum>\.CHECKSUM)?$"
)
_FUNDING: Final = re.compile(
    rf"^{re.escape(VISION_PREFIX)}futures/um/monthly/fundingRate/(?P<symbol>{_SYMBOL})"
    rf"/(?P<file_symbol>{_SYMBOL})-fundingRate-(?P<period>[0-9]{{4}}-[0-9]{{2}})"
    r"\.zip(?P<checksum>\.CHECKSUM)?$"
)


class UnrecognizedArchiveUrlError(ValueError):
    """The URL is not a recognised Vision archive (or checksum) path."""


class ArchiveSpecMismatchError(ValueError):
    """A declared ArchiveSpec disagrees with the URL it points at."""


@dataclass(frozen=True, slots=True)
class ParsedArchiveUrl:
    kind: ArchiveKind
    market: StoreMarket
    symbol: str
    interval: str | None
    period_kind: PeriodKind
    period_start: datetime
    period_end: datetime
    is_checksum: bool


def _period(text: str, cadence: str, url: str) -> tuple[datetime, datetime]:
    try:
        if cadence == "monthly":
            if len(text) != 7:
                raise ValueError("monthly archives are named YYYY-MM")
            start = datetime.strptime(text, "%Y-%m").replace(tzinfo=UTC)
            following = (start + timedelta(days=32)).replace(day=1)
            return start, following
        if len(text) != 10:
            raise ValueError("daily archives are named YYYY-MM-DD")
        start = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
        return start, start + timedelta(days=1)
    except ValueError as error:
        raise UnrecognizedArchiveUrlError(f"invalid archive period in {url}") from error


def parse_archive_url(url: str) -> ParsedArchiveUrl:
    """Parse a Vision archive or ``.CHECKSUM`` URL; refuse anything unrecognised."""

    if not isinstance(url, str) or not url.startswith(VISION_PREFIX):
        raise UnrecognizedArchiveUrlError("not an official Binance Vision URL")
    match = _KLINES.match(url)
    if match is not None:
        if match["symbol"] != match["file_symbol"] or (
            match["interval"] != match["file_interval"]
        ):
            raise UnrecognizedArchiveUrlError(
                f"directory and file name disagree in {url}"
            )
        start, end = _period(match["period"], match["cadence"], url)
        return ParsedArchiveUrl(
            kind=ArchiveKind.KLINES,
            market=StoreMarket.SPOT if match["root"] == "spot" else StoreMarket.USDS_M,
            symbol=match["symbol"],
            interval=match["interval"],
            period_kind=PeriodKind(match["cadence"]),
            period_start=start,
            period_end=end,
            is_checksum=match["checksum"] is not None,
        )
    match = _FUNDING.match(url)
    if match is not None:
        if match["symbol"] != match["file_symbol"]:
            raise UnrecognizedArchiveUrlError(
                f"directory and file name disagree in {url}"
            )
        start, end = _period(match["period"], "monthly", url)
        return ParsedArchiveUrl(
            kind=ArchiveKind.FUNDING_RATE,
            market=StoreMarket.USDS_M,
            symbol=match["symbol"],
            interval=None,
            period_kind=PeriodKind.MONTHLY,
            period_start=start,
            period_end=end,
            is_checksum=match["checksum"] is not None,
        )
    raise UnrecognizedArchiveUrlError(f"unrecognised Binance Vision URL: {url}")


def verify_spec_matches_url(spec: ArchiveSpec) -> ParsedArchiveUrl:
    """Parse ``spec.url`` and ``spec.checksum_url``; refuse any disagreement."""

    parsed = parse_archive_url(spec.url)
    checksum = parse_archive_url(spec.checksum_url)
    if parsed.is_checksum or not checksum.is_checksum:
        raise ArchiveSpecMismatchError(
            f"archive/checksum URLs are swapped or malformed: {spec.url}"
        )
    declared = (
        spec.kind,
        spec.market,
        spec.symbol,
        spec.interval,
        spec.period_kind,
        spec.period_start,
        spec.period_end,
    )
    from_url = (
        parsed.kind,
        parsed.market,
        parsed.symbol,
        parsed.interval,
        parsed.period_kind,
        parsed.period_start,
        parsed.period_end,
    )
    if declared != from_url or parsed != _without_checksum_flag(checksum):
        raise ArchiveSpecMismatchError(
            f"declared archive metadata disagrees with its URL {spec.url}: "
            f"declared {declared!r}, URL says {from_url!r}"
        )
    return parsed


def _without_checksum_flag(parsed: ParsedArchiveUrl) -> ParsedArchiveUrl:
    return ParsedArchiveUrl(
        parsed.kind,
        parsed.market,
        parsed.symbol,
        parsed.interval,
        parsed.period_kind,
        parsed.period_start,
        parsed.period_end,
        False,
    )


def check_url_fetchable(url: str, policy: EmbargoPolicy) -> ParsedArchiveUrl:
    """Parse ``url`` and apply ``policy`` to the period the URL itself names."""

    parsed = parse_archive_url(url)
    policy.check_archive_end(parsed.period_end, url=url)
    return parsed


__all__ = [
    "VISION_PREFIX",
    "ArchiveSpecMismatchError",
    "ParsedArchiveUrl",
    "UnrecognizedArchiveUrlError",
    "check_url_fetchable",
    "parse_archive_url",
    "verify_spec_matches_url",
]
