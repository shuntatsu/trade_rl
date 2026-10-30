"""Deterministic Vision-archive fixtures, a fake fetcher and an in-memory store.

Nothing here touches the network, a clock or PostgreSQL. :class:`MemoryStore`
reproduces the semantics ingestion relies on (primary keys, append-only rows,
rollback of a failed archive transaction) so the orchestration is exercised without a
database; the PostgreSQL integration tests then prove the same behaviour for real.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from trade_rl.integrations.market_store.checksum import ChecksumEvidence
from trade_rl.integrations.market_store.contracts import (
    ArchiveSpec,
    EmbargoState,
    FundingRow,
    KlineRow,
    StoredExchangeInfo,
    StoreMarket,
    ms_from_utc,
    utc_from_ms,
)
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    SEALED_WINDOW_START,
    EmbargoPolicy,
)
from trade_rl.integrations.market_store.ingest import (
    ConflictRecord,
    FetchResult,
    RunCounts,
    SourceRecord,
)
from trade_rl.integrations.market_store.plan import IngestionPlan

HOUR_MS = 3_600_000
SYMBOLS = ("BTCUSDT", "ETHUSDT")
NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)

EXCHANGE_INFO: dict[str, dict[str, object]] = {
    "usds_m": {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "onboardDate": 1_569_398_400_000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
            for symbol in SYMBOLS
        ]
    },
    "spot": {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                ],
            }
            for symbol in SYMBOLS
        ]
    },
}


def exchange_info_bytes(market: StoreMarket) -> bytes:
    return json.dumps(EXCHANGE_INFO[market.value], sort_keys=True).encode("utf-8")


def stored_exchange_info(market: StoreMarket) -> StoredExchangeInfo:
    raw = exchange_info_bytes(market)
    return StoredExchangeInfo(
        market=market,
        raw_payload=raw,
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
        source_uri=f"fixture://exchange-info/{market.value}",
        retrieved_at=datetime(2026, 10, 29, 8, tzinfo=UTC),
    )


def _price(value: int) -> str:
    return f"{Decimal(value) / Decimal(100_000_000):.8f}"


def kline_csv_rows(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    interval_ms: int = HOUR_MS,
    header: bool = False,
    micro: bool = False,
    skip: frozenset[int] = frozenset(),
    close_bump: int = 0,
) -> list[list[str]]:
    """Deterministic regular klines in ``[start_ms, end_ms)`` as Vision CSV rows."""

    seed = sum(ord(char) for char in symbol)
    price = 100_000_000_00 + seed * 1_000_000
    rows: list[list[str]] = []
    if header:
        rows.append(
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]
        )
    index = 0
    for open_ms in range(start_ms, end_ms, interval_ms):
        step = ((index * 7919 + seed) % 2001) - 1000
        open_price = price
        close_price = max(1_000_000, price + step * 10_000)
        high = max(open_price, close_price) + (index % 13) * 5_000 + 1_000
        low = min(open_price, close_price) - (index % 11) * 5_000 - 1_000
        low = max(low, 500_000)
        price = close_price
        index += 1
        if open_ms in skip:
            continue
        volume = 1000 + (index % 17) * 13
        quote = Decimal(volume) * Decimal(close_price) / Decimal(100_000_000)
        scale = 1_000 if micro else 1
        close_ms = open_ms + interval_ms - 1
        rows.append(
            [
                str(open_ms * scale),
                _price(open_price),
                _price(high),
                _price(low),
                _price(close_price + close_bump),
                f"{Decimal(volume) / Decimal(7):.8f}",
                str(close_ms * scale + (scale - 1)),
                f"{quote:.8f}",
                str(100 + index % 400),
                f"{Decimal(volume) / Decimal(14):.8f}",
                f"{quote / 2:.8f}",
                "0",
            ]
        )
    return rows


def funding_csv_rows(
    start_ms: int, end_ms: int, *, header: str = "calc_time"
) -> list[list[str]]:
    names = (
        ["calc_time", "funding_interval_hours", "last_funding_rate"]
        if header == "calc_time"
        else ["fundingTime", "fundingRate", "markPrice"]
    )
    rows = [names]
    index = 0
    for funding_ms in range(start_ms, end_ms, 8 * HOUR_MS):
        rate = f"{Decimal(index % 9 - 4) / Decimal(100_000):.8f}"
        stamp = str(funding_ms + 5)
        if header == "calc_time":
            rows.append([stamp, "8", rate])
        else:
            rows.append([stamp, rate, "43000.12000000"])
        index += 1
    return rows


def zip_csv(rows: Sequence[Sequence[str]], *, member: str = "data.csv") -> bytes:
    text = "\n".join(",".join(row) for row in rows) + "\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def checksum_file(payload: bytes, filename: str) -> bytes:
    return f"{hashlib.sha256(payload).hexdigest()}  {filename}\n".encode()


class FakeFetcher:
    """Serves prepared archives and checksums; records every requested URL."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.statuses: dict[str, int] = {}
        self.requested: list[str] = []

    def add(self, url: str, archive: bytes, *, checksum: bytes | None = None) -> None:
        self.files[url] = archive
        self.files[f"{url}.CHECKSUM"] = (
            checksum
            if checksum is not None
            else checksum_file(archive, url.rsplit("/", 1)[1])
        )

    def fail(self, url: str, status: int) -> None:
        self.statuses[url] = status

    def fetch(self, url: str) -> FetchResult:
        self.requested.append(url)
        if url in self.statuses:
            return FetchResult(status=self.statuses[url], body=None)
        body = self.files.get(url)
        if body is None:
            return FetchResult(status=404, body=None)
        return FetchResult(status=200, body=body)


def archive_bytes(spec: ArchiveSpec, **kwargs: object) -> bytes:
    """A valid archive for ``spec`` covering exactly its period."""

    start_ms = ms_from_utc(spec.period_start)
    end_ms = ms_from_utc(spec.period_end)
    if spec.interval is None:
        return zip_csv(funding_csv_rows(start_ms, end_ms))
    return zip_csv(
        kline_csv_rows(spec.symbol, start_ms, end_ms, **kwargs)  # type: ignore[arg-type]
    )


def fetcher_for(plan: IngestionPlan) -> FakeFetcher:
    fetcher = FakeFetcher()
    for spec in plan.specs:
        fetcher.add(spec.url, archive_bytes(spec))
    return fetcher


def populate_vision_cache(
    cache_root: Path, fetcher: FakeFetcher, plan: IngestionPlan
) -> None:
    """Lay the plan's archives into the existing Vision transport cache layout."""

    from trade_rl.integrations.binance import vision_cache_path

    for spec in plan.specs:
        payload = fetcher.files[spec.url]
        destination = vision_cache_path(cache_root, spec.url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        evidence = {
            "acquired_at": "2026-10-29T00:00:00+00:00",
            "downloader": "tests/market_store",
            "etag": None,
            "last_modified": None,
            "schema_version": "binance_vision_raw_cache_v1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "url": spec.url,
        }
        destination.with_suffix(".json").write_text(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def write_frozen_exchange_info(root: Path, market: StoreMarket) -> None:
    info = stored_exchange_info(market)
    root.mkdir(parents=True, exist_ok=True)
    (root / "exchange-info.raw.json").write_bytes(info.raw_payload)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "market": market.binance.value,
                "raw_payload_sha256": info.raw_payload_sha256,
                "retrieved_at": info.retrieved_at.isoformat(),
                "schema_version": "frozen_metadata_cache_v1",
                "source_uri": info.source_uri,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


class _Rollback(Exception):
    pass


class MemoryStore:
    """In-memory :class:`IngestStore` and :class:`MarketReader` with DB semantics."""

    def __init__(self) -> None:
        self.instruments: dict[tuple[StoreMarket, str], int] = {}
        self.sources: dict[str, dict[str, object]] = {}
        self.klines: dict[tuple[int, str, int], KlineRow] = {}
        self.funding: dict[tuple[int, int], FundingRow] = {}
        self.runs: dict[int, dict[str, object]] = {}
        self.conflicts: list[ConflictRecord] = []
        self.exchange_info: list[StoredExchangeInfo] = []
        self._next_source = 1

    # -- IngestStore ----------------------------------------------------------

    def begin_run(
        self,
        *,
        code_commit: str,
        command_line: str,
        request: Mapping[str, object],
        embargo: EmbargoPolicy,
    ) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = {
            "status": "running",
            "code_commit": code_commit,
            "command_line": command_line,
            "fs1_warmup_allowed": embargo.warmup_allowed,
            "fs1_warmup_authorization_url": embargo.warmup_authorization_url,
        }
        return run_id

    def finish_run(
        self, run_id: int, *, status: str, counts: RunCounts, error: str | None
    ) -> None:
        self.runs[run_id].update(status=status, counts=counts.to_dict(), error=error)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        saved = (dict(self.sources), dict(self.klines), dict(self.funding))
        try:
            yield
        except BaseException:
            self.sources, self.klines, self.funding = saved
            raise

    def ensure_instrument(self, market: StoreMarket, symbol: str) -> int:
        return self.instruments.setdefault((market, symbol), len(self.instruments) + 1)

    def source_record(self, url: str) -> SourceRecord | None:
        row = self.sources.get(url)
        if row is None:
            return None
        return SourceRecord(str(row["status"]), row["sha256"])  # type: ignore[arg-type]

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
    ) -> int:
        existing = self.sources.get(spec.url)
        if existing is not None and existing["status"] != "missing":
            raise AssertionError("an ingested source is immutable")
        source_id = (
            int(existing["source_id"])  # type: ignore[call-overload]
            if existing is not None
            else self._next_source
        )
        if existing is None:
            self._next_source += 1
        self.sources[spec.url] = {
            "source_id": source_id,
            "status": "missing" if evidence is None else "ingested",
            "http_status": http_status,
            "sha256": None if evidence is None else evidence.archive_sha256,
            "checksum_file_sha256": None
            if evidence is None
            else evidence.checksum_file_sha256,
            "size_bytes": None if evidence is None else evidence.size_bytes,
            "row_count": row_count,
            "period_end": spec.period_end,
        }
        return source_id

    def existing_klines(
        self, instrument_id: int, interval: str, start_ms: int, end_ms: int
    ) -> dict[int, KlineRow]:
        return {
            key[2]: row
            for key, row in self.klines.items()
            if key[0] == instrument_id
            and key[1] == interval
            and start_ms <= key[2] < end_ms
        }

    def insert_klines(
        self,
        instrument_id: int,
        interval: str,
        source_id: int,
        rows: Sequence[KlineRow],
    ) -> int:
        for row in rows:
            key = (instrument_id, interval, row.open_time_ms)
            if key in self.klines:
                raise AssertionError("primary key violation")
            self.klines[key] = row
        return len(rows)

    def existing_funding(
        self, instrument_id: int, start_ms: int, end_ms: int
    ) -> dict[int, FundingRow]:
        return {
            key[1]: row
            for key, row in self.funding.items()
            if key[0] == instrument_id and start_ms <= key[1] < end_ms
        }

    def insert_funding(
        self, instrument_id: int, source_id: int, rows: Sequence[FundingRow]
    ) -> int:
        for row in rows:
            key = (instrument_id, row.funding_time_ms)
            if key in self.funding:
                raise AssertionError("primary key violation")
            self.funding[key] = row
        return len(rows)

    def record_conflicts(
        self, run_id: int, conflicts: Sequence[ConflictRecord]
    ) -> None:
        self.conflicts.extend(conflicts)

    # -- MarketReader ---------------------------------------------------------

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        instrument_id = self.instruments.get((market, symbol))
        if instrument_id is None:
            return []
        rows = self.existing_klines(instrument_id, interval, start_ms, end_ms)
        return [rows[key] for key in sorted(rows)]

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        instrument_id = self.instruments.get((market, symbol))
        if instrument_id is None:
            return []
        rows = {
            key[1]: row
            for key, row in self.funding.items()
            if key[0] == instrument_id and start_ms <= key[1] <= end_ms
        }
        return [rows[key] for key in sorted(rows)]

    def load_exchange_info(
        self, market: StoreMarket, sha256: str | None = None
    ) -> StoredExchangeInfo:
        candidates = [
            info
            for info in self.exchange_info
            if info.market is market
            and (sha256 is None or info.raw_payload_sha256 == sha256)
        ]
        if not candidates:
            raise LookupError("no exchangeInfo snapshot")
        return max(candidates, key=lambda info: info.retrieved_at)

    def embargo_state(self) -> EmbargoState:
        return EmbargoState(
            fs1_warmup_start=FS1_WARMUP_START, sealed_from=SEALED_WINDOW_START
        )


def month_after(value: datetime) -> datetime:
    return (value.replace(day=1) + timedelta(days=32)).replace(day=1)


__all__ = [
    "HOUR_MS",
    "NOW",
    "SYMBOLS",
    "FakeFetcher",
    "MemoryStore",
    "archive_bytes",
    "checksum_file",
    "exchange_info_bytes",
    "fetcher_for",
    "file_backed_build",
    "funding_csv_rows",
    "kline_csv_rows",
    "month_after",
    "populate_vision_cache",
    "stored_exchange_info",
    "utc_from_ms",
    "write_frozen_exchange_info",
    "zip_csv",
]


def file_backed_build(  # type: ignore[no-untyped-def]
    root: Path,
    fetcher: FakeFetcher,
    plan: IngestionPlan,
    *,
    market: StoreMarket,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    evidence: Mapping[str, object],
    feature_timeframes: Sequence[str] = ("4h", "1d"),
):
    """The pre-existing file-backed path: Vision cache + frozen exchange info."""

    from trade_rl.data.build.economics import ExecutionEconomicsProfile
    from trade_rl.integrations.binance import (
        BinancePublicTransport,
        FrozenBinanceExchangeInfoTransport,
        build_binance_market_dataset,
    )
    from trade_rl.integrations.binance.dataset import BinanceGapPolicy
    from trade_rl.integrations.binance.derived_klines import (
        BaseIntervalAggregatingTransport,
    )

    cache = root / "vision-cache"
    populate_vision_cache(cache, fetcher, plan)
    write_frozen_exchange_info(root / "exchange-info", market)
    frozen = FrozenBinanceExchangeInfoTransport(root / "exchange-info", delegate=None)
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache, allow_network=False),
        exchange_information=frozen,
        base_interval="1h",
    )
    return build_binance_market_dataset(
        market=market.binance,
        symbols=tuple(symbols),
        interval="1h",
        start_time=start,
        end_time=end,
        transport_mode="vision",
        transport=transport,
        feature_timeframes=tuple(feature_timeframes),
        metadata_evidence=evidence,
        execution_economics=ExecutionEconomicsProfile(
            name="bot_overlay_only_assumptions_v1"
        ),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
