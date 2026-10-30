"""Idempotent, fail-closed ingestion of planned Vision archives into a store.

The orchestration depends only on the :class:`IngestStore` and :class:`ArchiveFetcher`
protocols, so it is unit-tested with in-memory fakes and no network or database.

Per archive:

1. a source already ingested is skipped without touching the network;
2. the archive and its official ``.CHECKSUM`` are fetched; a missing archive (404) is
   recorded as such and retried on the next run, any other failure aborts the run;
3. the archive is verified against the checksum, parsed, and every row is checked
   against the sealed-window embargo, all before the first write;
4. in one transaction the source row is written, existing rows for the same natural
   keys are compared by value, identical rows are counted (re-ingest is a no-op) and
   only unseen keys are inserted. A changed value for an existing key rolls the
   transaction back, is recorded as a conflict and aborts the run: nothing is ever
   overwritten.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from trade_rl.integrations.market_store.checksum import (
    ChecksumEvidence,
    verify_archive,
)
from trade_rl.integrations.market_store.contracts import (
    ArchiveKind,
    ArchiveSpec,
    FundingRow,
    KlineRow,
    StoreMarket,
    ms_from_utc,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.parse import (
    parse_funding_archive,
    parse_kline_archive,
)
from trade_rl.integrations.market_store.plan import IngestionPlan


class IngestError(RuntimeError):
    """The run cannot continue safely (fetch failure, vanished archive, ...)."""


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    """A changed value for an existing natural key (or a changed archive hash)."""

    kind: str
    market: StoreMarket
    symbol: str
    source_url: str
    bar_interval: str | None
    key_time_ms: int | None
    column: str
    existing: str | None
    incoming: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bar_interval": self.bar_interval,
            "column": self.column,
            "existing": self.existing,
            "incoming": self.incoming,
            "key_time_ms": self.key_time_ms,
            "kind": self.kind,
            "market": self.market.value,
            "source_url": self.source_url,
            "symbol": self.symbol,
        }


class ConflictError(IngestError):
    """Re-ingested data differs from stored data; nothing was overwritten."""

    def __init__(self, conflicts: Sequence[ConflictRecord]) -> None:
        self.conflicts = tuple(conflicts)
        first = self.conflicts[0]
        super().__init__(
            f"{len(self.conflicts)} conflicting value(s) for existing keys; first: "
            f"{first.kind} {first.symbol} {first.column} at {first.key_time_ms} "
            f"(stored {first.existing!r}, incoming {first.incoming!r}) in "
            f"{first.source_url}; nothing was overwritten"
        )


@dataclass(frozen=True, slots=True)
class FetchResult:
    status: int
    body: bytes | None


class ArchiveFetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


@dataclass(frozen=True, slots=True)
class SourceRecord:
    status: str
    sha256: str | None


@dataclass(slots=True)
class RunCounts:
    archives_planned: int = 0
    archives_ingested: int = 0
    archives_skipped: int = 0
    archives_missing: int = 0
    rows_inserted: int = 0
    rows_identical: int = 0
    conflicts: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "archives_ingested": self.archives_ingested,
            "archives_missing": self.archives_missing,
            "archives_planned": self.archives_planned,
            "archives_skipped": self.archives_skipped,
            "conflicts": self.conflicts,
            "rows_identical": self.rows_identical,
            "rows_inserted": self.rows_inserted,
        }


@dataclass(frozen=True, slots=True)
class IngestReport:
    run_id: int
    status: str
    counts: RunCounts
    missing_urls: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts.to_dict(),
            "missing_urls": list(self.missing_urls),
            "run_id": self.run_id,
            "status": self.status,
        }


class IngestStore(Protocol):
    """The writes ingestion needs; implemented by the PostgreSQL store and test fakes."""

    def begin_run(
        self,
        *,
        code_commit: str,
        command_line: str,
        request: Mapping[str, object],
        embargo: EmbargoPolicy,
    ) -> int: ...

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        counts: RunCounts,
        error: str | None,
    ) -> None: ...

    def transaction(self) -> AbstractContextManager[None]: ...

    def ensure_instrument(self, market: StoreMarket, symbol: str) -> int: ...

    def source_record(self, url: str) -> SourceRecord | None: ...

    def write_source(
        self,
        *,
        run_id: int,
        spec: ArchiveSpec,
        instrument_id: int,
        http_status: int,
        fetched_at: datetime,
        evidence: ChecksumEvidence | None,
        row_count: int | None,
    ) -> int: ...

    def existing_klines(
        self, instrument_id: int, interval: str, start_ms: int, end_ms: int
    ) -> dict[int, KlineRow]: ...

    def insert_klines(
        self,
        instrument_id: int,
        interval: str,
        source_id: int,
        rows: Sequence[KlineRow],
    ) -> int: ...

    def existing_funding(
        self, instrument_id: int, start_ms: int, end_ms: int
    ) -> dict[int, FundingRow]: ...

    def insert_funding(
        self, instrument_id: int, source_id: int, rows: Sequence[FundingRow]
    ) -> int: ...

    def record_conflicts(
        self, run_id: int, conflicts: Sequence[ConflictRecord]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class Diff[RowT]:
    new: tuple[tuple[int, RowT], ...]
    identical: int
    conflicts: tuple[tuple[int, str, str, str], ...]


def _text(value: object) -> str:
    return str(value)


def diff_rows[RowT: (KlineRow, FundingRow)](
    existing: Mapping[int, RowT],
    incoming: Sequence[tuple[int, RowT]],
) -> Diff[RowT]:
    """Classify incoming rows against stored rows by exact value equality.

    ``Decimal`` equality ignores trailing zeros, so ``1.50`` and ``1.5`` are the same
    value. Any other difference for an existing key is a conflict; the stored row is
    never modified.
    """

    new: list[tuple[int, RowT]] = []
    conflicts: list[tuple[int, str, str, str]] = []
    identical = 0
    for key, row in incoming:
        stored = existing.get(key)
        if stored is None:
            new.append((key, row))
            continue
        before = stored.comparable()
        after = row.comparable()
        changed = [name for name in after if before[name] != after[name]]
        if not changed:
            identical += 1
            continue
        conflicts.extend(
            (key, name, _text(before[name]), _text(after[name])) for name in changed
        )
    return Diff(tuple(new), identical, tuple(conflicts))


def _conflict_records(
    spec: ArchiveSpec, diff: Diff[KlineRow] | Diff[FundingRow]
) -> list[ConflictRecord]:
    kind = "kline" if spec.kind is ArchiveKind.KLINES else "funding_rate"
    return [
        ConflictRecord(
            kind=kind,
            market=spec.market,
            symbol=spec.symbol,
            source_url=spec.url,
            bar_interval=spec.interval,
            key_time_ms=key,
            column=column,
            existing=before,
            incoming=after,
        )
        for key, column, before, after in diff.conflicts
    ]


class _Refused(Exception):
    """Internal: unwinds the per-archive transaction so nothing is committed."""

    def __init__(self, conflicts: list[ConflictRecord]) -> None:
        super().__init__("conflict")
        self.conflicts = conflicts


def _ingest_archive(
    store: IngestStore,
    fetcher: ArchiveFetcher,
    spec: ArchiveSpec,
    *,
    run_id: int,
    embargo: EmbargoPolicy,
    refetch: bool,
    clock: Callable[[], datetime],
    counts: RunCounts,
    missing: list[str],
) -> None:
    embargo.check_archive_end(spec.period_end, url=spec.url)
    instrument_id = store.ensure_instrument(spec.market, spec.symbol)
    known = store.source_record(spec.url)
    if known is not None and known.status == "ingested" and not refetch:
        counts.archives_skipped += 1
        return

    fetched_at = clock()
    archive = fetcher.fetch(spec.url)
    if archive.status == 404:
        if known is not None and known.status == "ingested":
            raise IngestError(f"previously ingested archive is now missing: {spec.url}")
        with store.transaction():
            store.write_source(
                run_id=run_id,
                spec=spec,
                instrument_id=instrument_id,
                http_status=404,
                fetched_at=fetched_at,
                evidence=None,
                row_count=None,
            )
        counts.archives_missing += 1
        missing.append(spec.url)
        return
    if archive.status != 200 or not archive.body:
        raise IngestError(
            f"archive fetch failed with HTTP {archive.status}: {spec.url}"
        )
    checksum = fetcher.fetch(spec.checksum_url)
    if checksum.status != 200 or not checksum.body:
        raise IngestError(
            f"official checksum unavailable (HTTP {checksum.status}): {spec.checksum_url}"
        )
    evidence = verify_archive(archive.body, checksum.body, filename=spec.filename)

    if known is not None and known.status == "ingested":
        if known.sha256 == evidence.archive_sha256:
            counts.archives_skipped += 1
            return
        record = ConflictRecord(
            kind="archive_sha256",
            market=spec.market,
            symbol=spec.symbol,
            source_url=spec.url,
            bar_interval=spec.interval,
            key_time_ms=None,
            column="sha256",
            existing=known.sha256,
            incoming=evidence.archive_sha256,
        )
        counts.conflicts += 1
        store.record_conflicts(run_id, [record])
        raise ConflictError([record])

    start_ms = ms_from_utc(spec.period_start)
    end_ms = ms_from_utc(spec.period_end)
    kline_rows: list[KlineRow] = []
    funding_rows: list[FundingRow] = []
    if spec.kind is ArchiveKind.KLINES:
        kline_rows = parse_kline_archive(archive.body, spec, embargo=embargo)
        row_count = len(kline_rows)
    else:
        funding_rows = parse_funding_archive(archive.body, spec, embargo=embargo)
        row_count = len(funding_rows)

    try:
        with store.transaction():
            source_id = store.write_source(
                run_id=run_id,
                spec=spec,
                instrument_id=instrument_id,
                http_status=200,
                fetched_at=fetched_at,
                evidence=evidence,
                row_count=row_count,
            )
            if spec.kind is ArchiveKind.KLINES:
                assert spec.interval is not None
                stored_klines = store.existing_klines(
                    instrument_id, spec.interval, start_ms, end_ms
                )
                kline_diff = diff_rows(
                    stored_klines, [(row.open_time_ms, row) for row in kline_rows]
                )
                if kline_diff.conflicts:
                    raise _Refused(_conflict_records(spec, kline_diff))
                inserted = store.insert_klines(
                    instrument_id,
                    spec.interval,
                    source_id,
                    [row for _, row in kline_diff.new],
                )
                identical = kline_diff.identical
            else:
                stored_funding = store.existing_funding(instrument_id, start_ms, end_ms)
                funding_diff = diff_rows(
                    stored_funding,
                    [(row.funding_time_ms, row) for row in funding_rows],
                )
                if funding_diff.conflicts:
                    raise _Refused(_conflict_records(spec, funding_diff))
                inserted = store.insert_funding(
                    instrument_id, source_id, [row for _, row in funding_diff.new]
                )
                identical = funding_diff.identical
    except _Refused as refused:
        counts.conflicts += len(refused.conflicts)
        store.record_conflicts(run_id, refused.conflicts)
        raise ConflictError(refused.conflicts) from None
    counts.archives_ingested += 1
    counts.rows_inserted += inserted
    counts.rows_identical += identical


def run_ingestion(
    store: IngestStore,
    fetcher: ArchiveFetcher,
    plan: IngestionPlan,
    *,
    code_commit: str,
    command_line: str,
    refetch: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> IngestReport:
    """Ingest every planned archive; the run record is finished even on failure."""

    now = clock or (lambda: datetime.now(UTC))
    run_id = store.begin_run(
        code_commit=code_commit,
        command_line=command_line,
        request=plan.to_dict() | {"refetch": refetch},
        embargo=plan.embargo,
    )
    counts = RunCounts(archives_planned=len(plan.specs))
    missing: list[str] = []
    try:
        for spec in plan.specs:
            _ingest_archive(
                store,
                fetcher,
                spec,
                run_id=run_id,
                embargo=plan.embargo,
                refetch=refetch,
                clock=now,
                counts=counts,
                missing=missing,
            )
    except BaseException as error:
        store.finish_run(
            run_id,
            status="failed",
            counts=counts,
            error=f"{type(error).__name__}: {error}"[:2000],
        )
        raise
    store.finish_run(run_id, status="succeeded", counts=counts, error=None)
    return IngestReport(
        run_id=run_id,
        status="succeeded",
        counts=counts,
        missing_urls=tuple(missing),
    )


__all__ = [
    "ArchiveFetcher",
    "ConflictError",
    "ConflictRecord",
    "Diff",
    "FetchResult",
    "IngestError",
    "IngestReport",
    "IngestStore",
    "RunCounts",
    "SourceRecord",
    "diff_rows",
    "run_ingestion",
]
