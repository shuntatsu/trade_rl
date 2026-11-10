"""Checksum-verified row source and pair loader for the multi-symbol carry screen.

Two interchangeable row sources feed one pair loader:

* :class:`VisionArchiveSource` needs no database and no Docker. It fetches the planned
  Binance Vision archives and their official ``.CHECKSUM`` files through an injected
  :class:`~trade_rl.integrations.market_store.ingest.ArchiveFetcher`, verifies every
  archive against its checksum, parses it with the market store's exact parser (which
  refuses the FS1 warm-up span and sealed window by itself) and records per-archive
  evidence. :class:`ArchiveDirFetcher` is the write-once on-disk archive cache.
* ``PostgresMarketStore`` already satisfies :class:`CarryRowSource`, so a populated
  store can be used unchanged.

Neither source reads a price into a decision; :func:`load_carry_pair` only converts
rows into the unchanged ``RawMarketSeries`` contract. A missing archive, an
incomplete period or an unexplained hourly gap raises
:class:`CarryCoverageError`; a failed checksum or malformed archive raises its own
error and is never turned into a coverage exclusion.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    parse_carry_perp_rows,
    parse_carry_spot_rows,
)
from trade_rl.integrations.market_store.archive_url import VISION_PREFIX
from trade_rl.integrations.market_store.checksum import verify_archive
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    FundingRow,
    KlineRow,
    StoreMarket,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.ingest import ArchiveFetcher, FetchResult
from trade_rl.integrations.market_store.parse import (
    parse_funding_archive,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan

HOUR_MS = 3_600_000


class CarryRowSource(Protocol):
    """Read access the pair loader needs; Vision archives and PostgreSQL provide it."""

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        """Klines with ``start_ms <= open_time < end_ms``, ascending."""
        ...

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        """Funding rows with ``start_ms <= funding_time <= end_ms``, ascending."""
        ...


class ArchiveDirFetcher:
    """Write-once on-disk archive cache in front of an optional network fetcher.

    A cached file is returned as is; every consumer re-verifies it against its
    official checksum, so a corrupted or edited cache entry cannot be accepted. A
    miss without an inner fetcher is status 0 (unavailable offline), never a 404.
    """

    def __init__(self, directory: Path, inner: ArchiveFetcher | None = None) -> None:
        self._directory = directory
        self._inner = inner

    def _path(self, url: str) -> Path:
        if not url.startswith(VISION_PREFIX):
            raise ValueError(f"not a Binance Vision URL: {url}")
        relative = url[len(VISION_PREFIX) :]
        parts = relative.split("/")
        if not parts or any(part in {"", ".", ".."} or "\\" in part for part in parts):
            raise ValueError(f"unsafe archive path: {url}")
        return self._directory.joinpath(*parts)

    def fetch(self, url: str) -> FetchResult:
        path = self._path(url)
        if path.is_file():
            return FetchResult(status=200, body=path.read_bytes())
        if self._inner is None:
            return FetchResult(status=0, body=None)
        result = self._inner.fetch(url)
        if result.status == 200 and result.body:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".part")
            temporary.write_bytes(result.body)
            os.replace(temporary, path)
        return result


class VisionArchiveSource:
    """Rows from planned Vision archives, each verified against its checksum."""

    def __init__(
        self,
        fetcher: ArchiveFetcher,
        plan: IngestionPlan,
        *,
        embargo: EmbargoPolicy | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._embargo = plan.embargo if embargo is None else embargo
        self._specs: dict[tuple[ArchiveKind, StoreMarket, str], list[ArchiveSpec]] = {}
        for spec in plan.specs:
            self._specs.setdefault((spec.kind, spec.market, spec.symbol), []).append(
                spec
            )
        for specs in self._specs.values():
            specs.sort(key=lambda spec: spec.period_start)
        self._evidence: dict[str, dict[str, object]] = {}

    def evidence(self) -> list[dict[str, object]]:
        """Per-archive verification evidence for every archive read, URL-sorted."""
        return [self._evidence[url] for url in sorted(self._evidence)]

    def _selected(
        self,
        kind: ArchiveKind,
        market: StoreMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[ArchiveSpec]:
        specs = self._specs.get((kind, market, symbol), [])
        chosen = [
            spec
            for spec in specs
            if ms_from_utc(spec.period_start) < end_ms
            and ms_from_utc(spec.period_end) > start_ms
        ]
        cursor = start_ms
        for spec in chosen:
            if ms_from_utc(spec.period_start) > cursor:
                break
            cursor = max(cursor, ms_from_utc(spec.period_end))
        else:
            if chosen and cursor >= end_ms:
                return chosen
        raise CarryCoverageError(
            f"no planned {kind.value} archives cover {market.value} {symbol} for "
            f"[{start_ms}, {end_ms})"
        )

    def _payload(self, spec: ArchiveSpec) -> bytes:
        archive = self._fetcher.fetch(spec.url)
        if archive.status == 404:
            raise CarryCoverageError(f"archive is not published: {spec.url}")
        if archive.status != 200 or not archive.body:
            raise CarryCoverageError(
                f"archive unavailable (HTTP {archive.status}): {spec.url}"
            )
        checksum = self._fetcher.fetch(spec.checksum_url)
        if checksum.status != 200 or not checksum.body:
            raise CarryCoverageError(
                f"official checksum unavailable (HTTP {checksum.status}): "
                f"{spec.checksum_url}"
            )
        proof = verify_archive(archive.body, checksum.body, filename=spec.filename)
        self._evidence[spec.url] = {
            "archive_sha256": proof.archive_sha256,
            "checksum_file_sha256": proof.checksum_file_sha256,
            "checksum_url": spec.checksum_url,
            "kind": spec.kind.value,
            "market": spec.market.value,
            "size_bytes": proof.size_bytes,
            "symbol": spec.symbol,
            "url": spec.url,
        }
        return archive.body

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        rows: list[KlineRow] = []
        for spec in self._selected(
            ArchiveKind.KLINES, market, symbol, start_ms, end_ms
        ):
            if spec.interval != interval:
                raise CarryCoverageError(f"planned interval differs: {spec.url}")
            parsed = parse_kline_archive(
                self._payload(spec), spec, embargo=self._embargo
            )
            rows.extend(
                row for row in parsed.rows if start_ms <= row.open_time_ms < end_ms
            )
        return rows

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        rows: list[FundingRow] = []
        for spec in self._selected(
            ArchiveKind.FUNDING_RATE, market, symbol, start_ms, end_ms
        ):
            rows.extend(
                row
                for row in parse_funding_archive(
                    self._payload(spec), spec, embargo=self._embargo
                )
                if start_ms <= row.funding_time_ms <= end_ms
            )
        return rows


def _vision_rows(rows: Sequence[KlineRow]) -> list[Sequence[object]]:
    return [row.vision_row() for row in rows]


def load_carry_pair(
    source: CarryRowSource,
    symbol: str,
    *,
    start_time: datetime,
    end_time: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
    interval: str = "1h",
) -> tuple[RawMarketSeries, RawMarketSeries]:
    """The (spot, perpetual) ``RawMarketSeries`` pair for one USDT symbol."""
    start_ms, end_ms = ms_from_utc(start_time), ms_from_utc(end_time)
    spot_rows = source.read_klines(StoreMarket.SPOT, symbol, interval, start_ms, end_ms)
    perp_rows = source.read_klines(
        StoreMarket.USDS_M, symbol, interval, start_ms, end_ms
    )
    funding = source.read_funding(StoreMarket.USDS_M, symbol, start_ms, end_ms)
    events = [(row.funding_time_ms, float(row.rate)) for row in funding]
    try:
        spot = parse_carry_spot_rows(
            _vision_rows(spot_rows),
            start_time=start_time,
            end_time=end_time,
            halt_intervals=halt_intervals,
        )
        perp = parse_carry_perp_rows(
            _vision_rows(perp_rows), events, start_time=start_time, end_time=end_time
        )
    except ValueError as error:
        raise CarryCoverageError(f"{symbol}: {error}") from error
    return spot, perp


def hourly_quote_volume(
    source: CarryRowSource,
    market: StoreMarket,
    symbol: str,
    *,
    start_time: datetime,
    end_time: datetime,
    interval: str = "1h",
) -> np.ndarray:
    """Complete hourly quote turnover; any missing hour is a coverage failure.

    Only the quote volume column is read, for the pre-registered liquidity proxy.
    """
    start_ms, end_ms = ms_from_utc(start_time), ms_from_utc(end_time)
    rows = source.read_klines(market, symbol, interval, start_ms, end_ms)
    expected = (end_ms - start_ms) // HOUR_MS
    opens = np.asarray([row.open_time_ms for row in rows], dtype=np.int64)
    if len(rows) != expected or not np.array_equal(
        opens, np.arange(start_ms, end_ms, HOUR_MS, dtype=np.int64)
    ):
        raise CarryCoverageError(
            f"{market.value} {symbol}: {len(rows)} of {expected} hourly bars present "
            "in the liquidity lookback"
        )
    return np.asarray([float(row.quote_volume) for row in rows], dtype=np.float64)


class CarryScreenData:
    """Row source -> the data interface of the multi-symbol carry screen runner.

    The runner owns the economics and imports nothing from this package; this class
    satisfies its ``ScreenData`` protocol structurally. It only reads timestamps and
    quote volume for the liquidity proxy and converts rows into raw series.
    """

    def __init__(
        self,
        source: CarryRowSource,
        *,
        lookback_start: datetime,
        window_start: datetime,
        window_end: datetime,
        halt_intervals: Sequence[tuple[datetime, datetime]],
    ) -> None:
        self._source = source
        self._lookback_start = lookback_start
        self._window_start = window_start
        self._window_end = window_end
        self._halts = tuple(halt_intervals)

    @property
    def name(self) -> str:
        return type(self._source).__name__

    def lookback_median_volumes(self, symbol: str) -> dict[str, float]:
        return {
            market.value: float(
                np.median(
                    hourly_quote_volume(
                        self._source,
                        market,
                        symbol,
                        start_time=self._lookback_start,
                        end_time=self._window_start,
                    )
                )
            )
            for market in (StoreMarket.SPOT, StoreMarket.USDS_M)
        }

    def pair(self, symbol: str) -> tuple[RawMarketSeries, RawMarketSeries]:
        return load_carry_pair(
            self._source,
            symbol,
            start_time=self._window_start,
            end_time=self._window_end,
            halt_intervals=self._halts,
        )

    def evidence(self) -> list[dict[str, object]]:
        evidence = getattr(self._source, "evidence", None)
        return list(evidence()) if callable(evidence) else []


__all__ = [
    "ArchiveDirFetcher",
    "CarryCoverageError",
    "CarryRowSource",
    "CarryScreenData",
    "VisionArchiveSource",
    "hourly_quote_volume",
    "load_carry_pair",
]
