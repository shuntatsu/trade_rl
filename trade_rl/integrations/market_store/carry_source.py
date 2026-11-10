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
rows into the unchanged ``RawMarketSeries`` contract. The failure taxonomy is strict.
Only a documented absence raises :class:`CarryCoverageError` (an HTTP 404 on a planned
archive, or a gap no declared halt explains) and may exclude a symbol, with its
evidence. Server errors, transport failures and timeouts are retried a bounded number
of times and then raise :class:`CarryOperationalHalt` (resumable); an offline cache
miss raises it at once. A checksum mismatch, a malformed archive and a duplicated,
unordered or malformed row raise :class:`CarryIntegrityHalt`. Halts never exclude.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data.source import RawMarketSeries
from trade_rl.integrations.binance.carry import (
    CarryCoverageError,
    CarryIntegrityHalt,
    CarryOperationalHalt,
    parse_carry_perp_rows,
    parse_carry_spot_rows,
)
from trade_rl.integrations.market_store.archive_url import VISION_PREFIX
from trade_rl.integrations.market_store.checksum import ChecksumError, verify_archive
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
    ArchiveContentError,
    parse_funding_archive,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan

HOUR_MS = 3_600_000
#: Status of a read that never reached the network: an offline cache miss.
OFFLINE_MISS_STATUS = -1
_TRANSIENT = frozenset({0, 418, 429})


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
    miss without an inner fetcher is ``OFFLINE_MISS_STATUS``, never a 404.
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

    @staticmethod
    def _read_cached(path: Path, url: str) -> bytes:
        # Open by checked descriptor to prevent a cache symlink swap after
        # Path.is_file() and before the actual cache read.
        try:
            with open_regular_binary(path, field="Vision archive cache") as handle:
                return handle.read()
        except (OSError, ValueError) as error:
            raise CarryOperationalHalt(
                f"archive cache entry could not be safely read: {url}",
                {"kind": "cache_read_failed", "url": url},
            ) from error

    def fetch(self, url: str) -> FetchResult:
        path = self._path(url)
        if path.is_symlink():
            raise CarryOperationalHalt(
                f"archive cache symlink is not trusted: {url}",
                {"kind": "cache_symlink", "url": url},
            )
        if path.is_file():
            return FetchResult(status=200, body=self._read_cached(path, url))
        if self._inner is None:
            return FetchResult(status=OFFLINE_MISS_STATUS, body=None)
        result = self._inner.fetch(url)
        if result.status == 200 and result.body:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Every publisher owns a private staging file. A hard-link claim is
            # exclusive even when a rival arrives after the initial cache miss;
            # os.replace would silently overwrite that rival on POSIX.
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=f".{path.name}.part-", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(result.body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, path)
            except FileExistsError as error:
                # Only the already-published cache winner can be returned.
                # VisionArchiveSource separately validates its official checksum.
                if not path.is_file() or path.is_symlink():
                    raise CarryOperationalHalt(
                        f"archive cache publication conflict: {url}",
                        {"kind": "cache_publication_conflict", "url": url},
                    ) from error
                return FetchResult(status=200, body=self._read_cached(path, url))
            except OSError as error:
                raise CarryOperationalHalt(
                    f"archive cache publication failed: {url}",
                    {"kind": "cache_publication_failed", "url": url},
                ) from error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return result


class VisionArchiveSource:
    """Rows from planned Vision archives, each verified against its checksum."""

    def __init__(
        self,
        fetcher: ArchiveFetcher,
        plan: IngestionPlan,
        *,
        embargo: EmbargoPolicy | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("attempts must be positive and backoff non-negative")
        self._fetcher = fetcher
        self._embargo = plan.embargo if embargo is None else embargo
        self._attempts = max_attempts
        self._backoff = backoff_seconds
        self._sleep = sleep
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
        # the plan itself does not cover the request: a defect, not a missing archive
        raise CarryIntegrityHalt(
            f"no planned {kind.value} archives cover {market.value} {symbol} for "
            f"[{start_ms}, {end_ms})",
            {
                "kind": "plan_does_not_cover",
                "market": market.value,
                "symbol": symbol,
            },
        )

    def _fetch(self, url: str) -> tuple[FetchResult, int]:
        """Fetch with a bounded retry of transient failures only (5xx, 429, status 0)."""
        for attempt in range(1, self._attempts + 1):
            result = self._fetcher.fetch(url)
            transient = result.status in _TRANSIENT or result.status >= 500
            if not transient or attempt == self._attempts:
                return result, attempt
            self._sleep(self._backoff * 2 ** (attempt - 1))
        raise AssertionError("unreachable")

    def _unavailable(
        self, url: str, result: FetchResult, attempts: int
    ) -> CarryOperationalHalt:
        if result.status == OFFLINE_MISS_STATUS:
            return CarryOperationalHalt(
                f"not in the offline archive cache: {url}",
                {
                    "kind": "offline_cache_miss",
                    "url": url,
                    "status": OFFLINE_MISS_STATUS,
                },
            )
        return CarryOperationalHalt(
            f"archive unavailable (HTTP {result.status}) after {attempts} attempt(s): "
            f"{url}",
            {
                "kind": "archive_unavailable",
                "url": url,
                "status": result.status,
                "attempts": attempts,
            },
        )

    def _payload(self, spec: ArchiveSpec) -> bytes:
        archive, attempts = self._fetch(spec.url)
        if archive.status == 404:
            raise CarryCoverageError(
                f"archive is not published: {spec.url}",
                {"kind": "archive_not_published", "url": spec.url, "status": 404},
            )
        if archive.status != 200:
            raise self._unavailable(spec.url, archive, attempts)
        checksum, attempts = self._fetch(spec.checksum_url)
        if checksum.status == 404:
            raise CarryIntegrityHalt(
                f"official checksum is not published: {spec.checksum_url}",
                {
                    "kind": "checksum_missing",
                    "url": spec.checksum_url,
                    "status": 404,
                },
            )
        if checksum.status != 200:
            raise self._unavailable(spec.checksum_url, checksum, attempts)
        try:
            proof = verify_archive(
                archive.body or b"", checksum.body or b"", filename=spec.filename
            )
        except ChecksumError as error:
            raise CarryIntegrityHalt(
                str(error),
                {"kind": "checksum_mismatch", "url": spec.url, "detail": str(error)},
            ) from error
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
        return archive.body or b""

    @staticmethod
    def _malformed(spec: ArchiveSpec, error: ArchiveContentError) -> CarryIntegrityHalt:
        return CarryIntegrityHalt(
            str(error),
            {"kind": "malformed_archive", "url": spec.url, "detail": str(error)},
        )

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
                raise CarryIntegrityHalt(
                    f"planned interval differs: {spec.url}",
                    {"kind": "plan_interval_mismatch", "url": spec.url},
                )
            payload = self._payload(spec)
            try:
                parsed = parse_kline_archive(payload, spec, embargo=self._embargo)
            except ArchiveContentError as error:
                raise self._malformed(spec, error) from error
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
            payload = self._payload(spec)
            try:
                parsed = parse_funding_archive(payload, spec, embargo=self._embargo)
            except ArchiveContentError as error:
                raise self._malformed(spec, error) from error
            rows.extend(
                row for row in parsed if start_ms <= row.funding_time_ms <= end_ms
            )
        return rows


def _vision_rows(rows: Sequence[KlineRow]) -> list[Sequence[object]]:
    return [row.vision_row() for row in rows]


def _check_order(
    opens: Sequence[int], *, market: StoreMarket, symbol: str, what: str
) -> None:
    """Duplicated or unordered rows are integrity faults, never coverage facts."""
    for index in range(1, len(opens)):
        if opens[index] <= opens[index - 1]:
            kind = (
                "duplicate_rows"
                if opens[index] == opens[index - 1]
                else "unordered_rows"
            )
            raise CarryIntegrityHalt(
                f"{market.value} {symbol}: {what} are not strictly increasing "
                f"at row {index}",
                {
                    "kind": kind,
                    "market": market.value,
                    "symbol": symbol,
                    "row": index,
                    "open_time_ms": int(opens[index]),
                },
            )


def _missing_hours(
    opens: Sequence[int],
    start_ms: int,
    end_ms: int,
    *,
    explained: Sequence[tuple[int, int]] = (),
) -> list[int]:
    """Hourly opens absent from ``opens`` that no declared halt fully contains.

    The very first bar can never be explained: a halt needs a preceding mark.
    """
    present = set(opens)
    gaps = []
    for index, open_ms in enumerate(range(start_ms, end_ms, HOUR_MS)):
        if open_ms in present:
            continue
        inside = index > 0 and any(
            lo <= open_ms and open_ms + HOUR_MS <= hi for lo, hi in explained
        )
        if not inside:
            gaps.append(open_ms)
    return gaps


def _gap_error(
    market: StoreMarket, symbol: str, gaps: Sequence[int]
) -> CarryCoverageError:
    return CarryCoverageError(
        f"{market.value} {symbol}: {len(gaps)} unexplained missing hourly bar(s), "
        f"the first opening at {gaps[0]}",
        {
            "kind": "gap",
            "market": market.value,
            "symbol": symbol,
            "first_missing_open_ms": gaps[0],
            "missing_hours": len(gaps),
        },
    )


def funding_coverage_gap(counts: np.ndarray) -> str | None:
    """The assembler's funding-coverage rule, as a reason (None when it passes).

    Mirrors ``assemble_carry_dataset`` so a funding gap is classified before assembly;
    a test pins the two together.
    """
    events = np.flatnonzero(counts)
    if not len(events):
        return "no funding events"
    if events[0] > 8:
        return "the first funding event is more than eight bars after the start"
    if len(counts) - 2 - events[-1] > 8:
        return "the last funding event is more than eight bars before the end"
    if np.any(np.diff(events) > 9):
        return "an interior funding gap exceeds nine hourly bars"
    return None


def load_carry_pair(
    source: CarryRowSource,
    symbol: str,
    *,
    start_time: datetime,
    end_time: datetime,
    halt_intervals: Sequence[tuple[datetime, datetime]],
    interval: str = "1h",
) -> tuple[RawMarketSeries, RawMarketSeries]:
    """The (spot, perpetual) ``RawMarketSeries`` pair for one USDT symbol.

    Row-level integrity is checked before gaps, and gaps before parsing, so a gap is
    the only thing that raises :class:`CarryCoverageError` here.
    """
    start_ms, end_ms = ms_from_utc(start_time), ms_from_utc(end_time)
    spot_rows = source.read_klines(StoreMarket.SPOT, symbol, interval, start_ms, end_ms)
    perp_rows = source.read_klines(
        StoreMarket.USDS_M, symbol, interval, start_ms, end_ms
    )
    funding = source.read_funding(StoreMarket.USDS_M, symbol, start_ms, end_ms)
    halts = [(ms_from_utc(lo), ms_from_utc(hi)) for lo, hi in halt_intervals]
    for market, rows, explained in (
        (StoreMarket.SPOT, spot_rows, halts),
        (StoreMarket.USDS_M, perp_rows, ()),
    ):
        opens = [row.open_time_ms for row in rows]
        _check_order(opens, market=market, symbol=symbol, what="klines")
        gaps = _missing_hours(opens, start_ms, end_ms, explained=explained)
        if gaps:
            raise _gap_error(market, symbol, gaps)
    stamps = [row.funding_time_ms for row in funding]
    _check_order(stamps, market=StoreMarket.USDS_M, symbol=symbol, what="funding")
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
        raise CarryIntegrityHalt(
            f"{symbol}: {error}",
            {"kind": "parse_failure", "symbol": symbol, "detail": str(error)},
        ) from error
    counts = np.asarray(perp.funding_event_count)
    if np.any(counts > 1):
        raise CarryIntegrityHalt(
            f"{symbol}: two funding events settle in one hourly bar",
            {"kind": "duplicate_funding", "symbol": symbol},
        )
    reason = funding_coverage_gap(counts)
    if reason is not None:
        raise CarryCoverageError(
            f"{symbol}: {reason}",
            {"kind": "funding_gap", "symbol": symbol, "detail": reason},
        )
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
    """Complete hourly quote turnover; a missing hour is a coverage fact.

    Only the quote volume column is read, for the pre-registered liquidity proxy.
    Duplicated or unordered rows are an integrity halt.
    """
    start_ms, end_ms = ms_from_utc(start_time), ms_from_utc(end_time)
    rows = source.read_klines(market, symbol, interval, start_ms, end_ms)
    opens = [row.open_time_ms for row in rows]
    _check_order(opens, market=market, symbol=symbol, what="lookback klines")
    gaps = _missing_hours(opens, start_ms, end_ms)
    if gaps:
        raise _gap_error(market, symbol, gaps)
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
    "OFFLINE_MISS_STATUS",
    "ArchiveDirFetcher",
    "CarryCoverageError",
    "CarryIntegrityHalt",
    "CarryOperationalHalt",
    "CarryRowSource",
    "CarryScreenData",
    "VisionArchiveSource",
    "funding_coverage_gap",
    "hourly_quote_volume",
    "load_carry_pair",
]
