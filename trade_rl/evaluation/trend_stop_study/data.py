"""Multi-symbol 4h / 1d bar Datasets for the trend study (offline, read-only).

Primary path: a published Dataset artifact written by the Market Store export CLI
(one export per timeframe; the command is documented in
``docs/specs/trend-following-stops.md``), loaded through the verified artifact reader.
Fallback: a
local Binance Vision cache of native 4h / 1d klines and funding archives, built with the
unchanged ``build_binance_market_dataset`` and never any network access.

Neither path reads a price into this module's logic: it only validates the roster, the
bar cadence and the hard data boundary. The roster is the eleven protocol symbols with
no exclusion path: a missing symbol is INVALID, never a smaller roster.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data import MarketDataset
from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.evaluation.trend_stop_study.protocol import TrendStopProtocol
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
    FrozenBinanceExchangeInfoTransport,
    inspect_binance_vision_cache,
    plan_binance_vision_cache,
    vision_cache_path,
)
from trade_rl.integrations.binance.cache import read_missing_archive_evidence
from trade_rl.integrations.binance.dataset import (
    BinanceGapPolicy,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)

_BAR_NS = {"4h": 4 * 3_600 * 10**9, "1d": 24 * 3_600 * 10**9}
ECONOMICS_PROFILE_NAME = "bot_overlay_only_assumptions_v1"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


# Every failure a data load can raise; the CLI maps each to INVALID or a halt.
DATA_LOAD_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    OSError,
    BinanceTransportError,
)


@dataclass(frozen=True)
class MissingArchive:
    """A roster archive the venue answered HTTP 404 for, with the recorded evidence."""

    symbol: str
    url: str
    http_status: int
    observed_at: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "url": self.url,
            "http_status": self.http_status,
            "observed_at": self.observed_at,
        }


class IncompleteRosterError(ValueError):
    """A roster symbol is missing: the run is INVALID, never a reduced roster.

    Raised for a Dataset that lacks a roster symbol and for a Vision cache that lacks
    an archive whose HTTP 404 is recorded (``archives``). It is terminal for this
    protocol: the caller writes an INVALID result and replays nothing.
    """

    def __init__(
        self, missing: Iterable[str], archives: Sequence[MissingArchive] = ()
    ) -> None:
        self.missing = tuple(sorted(missing))
        self.archives = tuple(archives)
        detail = ""
        if self.archives:
            detail = "; HTTP 404 recorded for " + ", ".join(
                f"{item.url} ({item.http_status} at {item.observed_at})"
                for item in self.archives
            )
        super().__init__(
            "roster symbols missing (no exclusions exist; the run is INVALID): "
            f"{list(self.missing)}{detail}"
        )


class OperationalDataError(ValueError):
    """A data failure that is not evidence of a missing archive: halt, write no result.

    The run is not consumed: the operator completes or repairs the cache and reruns.
    """


class IncompleteArchiveCacheError(OperationalDataError):
    """Archives are absent, empty or fail their content check, with no 404 on record."""

    def __init__(
        self,
        *,
        absent: Sequence[str],
        empty: Sequence[str],
        invalid: Sequence[str],
    ) -> None:
        self.absent = tuple(absent)
        self.empty = tuple(empty)
        self.invalid = tuple(invalid)
        shown = [*self.absent, *self.empty, *self.invalid][:5]
        super().__init__(
            "the Binance Vision cache is incomplete and this is NOT evidence of a "
            f"missing archive (absent={len(self.absent)} empty={len(self.empty)} "
            f"failed content check={len(self.invalid)}); no result was written. The "
            "operator must complete or repair the download (a recorded HTTP 404 "
            "would make the run INVALID instead) and rerun. First archives: "
            + ", ".join(shown)
        )


def load_exported_dataset(path: str | Path) -> MarketDataset:
    """Load and verify a published Dataset artifact (``<export-dir>/dataset``)."""

    return load_published_market_dataset_artifact(Path(path))[1]


def validate_study_dataset(
    dataset: MarketDataset, protocol: TrendStopProtocol, timeframe: str
) -> tuple[str, ...]:
    """Check roster, cadence, warm-up and the data boundary; return the symbols used.

    The roster is exactly ``protocol.symbols``: there is no exclusion path and no
    minimum roster size. A Dataset missing any roster symbol raises
    :class:`IncompleteRosterError` so the caller writes an INVALID result (no replay,
    no rejection of the rule); a symbol outside the roster is refused (``ValueError``).

    The boundary check is on the loaded data, not only on the requested window: a
    Dataset holding any bar that closes after ``data_end_exclusive`` is refused even if
    the evaluation window would not reach it.
    """

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if stamps.size < 2:
        raise ValueError("Dataset needs at least two bars")
    if int(stamps[-1]) > _ns(protocol.data_end_exclusive):
        raise ValueError(
            "Dataset holds bars after the exploratory data boundary "
            f"{protocol.data_end_exclusive}; refusing to load it"
        )
    steps = np.unique(np.diff(stamps))
    if steps.size != 1 or int(steps[0]) != _BAR_NS[timeframe]:
        raise ValueError(f"Dataset cadence is not a regular {timeframe} grid")
    if int(stamps[0]) > _ns(protocol.evaluation_start):
        raise ValueError("Dataset starts after the evaluation start (no warm-up)")
    if len(set(dataset.symbols)) != len(dataset.symbols):
        raise ValueError("Dataset holds duplicate symbols")
    unknown = sorted(set(dataset.symbols) - set(protocol.symbols))
    if unknown:
        raise ValueError(f"Dataset holds symbols outside the roster: {unknown}")
    missing = sorted(set(protocol.symbols) - set(dataset.symbols))
    if missing:
        raise IncompleteRosterError(missing)
    if protocol.benchmark_symbol not in dataset.symbols:
        raise ValueError("the benchmark symbol must be part of the Dataset")
    return tuple(dataset.symbols)


def plan_vision_cache(protocol: TrendStopProtocol, timeframe: str) -> Any:
    """The Vision archives a fallback build needs (pure; touches no network)."""

    return plan_binance_vision_cache(
        market=BinanceMarket.USDS_M,
        symbols=protocol.symbols,
        intervals=(timeframe,),
        start_time=_utc(protocol.data_start),
        end_time=_utc(protocol.data_end_exclusive),
    )


def _archive_symbol(protocol: TrendStopProtocol, url: str) -> str:
    parts = url.split("/")
    matches = [symbol for symbol in protocol.symbols if symbol in parts]
    if len(matches) != 1:
        raise OperationalDataError(f"cannot attribute the archive to one symbol: {url}")
    return matches[0]


def classify_vision_cache(
    protocol: TrendStopProtocol, timeframe: str, *, cache_root: Path
) -> None:
    """Return if the cache is complete; otherwise raise INVALID or an operational halt.

    * A missing archive with recorded HTTP 404 evidence (written by the download step
      when the venue answered 404) raises :class:`IncompleteRosterError`: INVALID and
      terminal. It takes precedence over any other gap.
    * Any other gap (not yet downloaded, empty, failed content check, unreadable
      evidence) raises :class:`IncompleteArchiveCacheError`: a resumable halt that
      writes no result.
    """

    plan = plan_vision_cache(protocol, timeframe)
    report = inspect_binance_vision_cache(plan, cache_root=cache_root)
    if report.complete:
        return
    evidenced: list[MissingArchive] = []
    absent: list[str] = []
    for url in report.missing_urls:
        try:
            record = read_missing_archive_evidence(
                vision_cache_path(cache_root, url), url
            )
        except BinanceTransportError:
            record = None  # unreadable evidence is not a 404: the operator repairs it
        if record is None:
            absent.append(url)
        else:
            evidenced.append(
                MissingArchive(
                    symbol=_archive_symbol(protocol, url),
                    url=url,
                    http_status=int(str(record["http_status"])),
                    observed_at=str(record["observed_at"]),
                )
            )
    if evidenced:
        raise IncompleteRosterError({item.symbol for item in evidenced}, evidenced)
    raise IncompleteArchiveCacheError(
        absent=absent, empty=report.empty_urls, invalid=report.invalid_urls
    )


def build_dataset_from_transport(
    protocol: TrendStopProtocol,
    timeframe: str,
    transport: Any,
    *,
    evidence: Mapping[str, object],
) -> MarketDataset:
    """Build with the unchanged Binance builder from any Vision-shaped transport."""

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    built = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=protocol.symbols,
        interval=timeframe,
        start_time=_utc(protocol.data_start),
        end_time=_utc(protocol.data_end_exclusive),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        metadata_evidence=dict(evidence),
        execution_economics=ExecutionEconomicsProfile(name=ECONOMICS_PROFILE_NAME),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
    validate_study_dataset(built.dataset, protocol, timeframe)
    return built.dataset


def build_vision_dataset(
    protocol: TrendStopProtocol,
    timeframe: str,
    *,
    cache_root: Path,
    exchange_info_dir: Path,
) -> MarketDataset:
    """Fallback: build from a complete local Vision cache; never uses the network."""

    if timeframe not in _BAR_NS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    classify_vision_cache(protocol, timeframe, cache_root=cache_root)
    frozen = FrozenBinanceExchangeInfoTransport(exchange_info_dir, delegate=None)
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache_root, allow_network=False),
        exchange_information=frozen,
        base_interval=timeframe,
    )
    return build_dataset_from_transport(
        protocol,
        timeframe,
        transport,
        evidence={
            "schema_version": "trend_stop_vision_fallback_v1",
            "timeframe": timeframe,
            "network": "never",
        },
    )


__all__ = [
    "DATA_LOAD_ERRORS",
    "IncompleteArchiveCacheError",
    "IncompleteRosterError",
    "MissingArchive",
    "OperationalDataError",
    "build_dataset_from_transport",
    "classify_vision_cache",
    "build_vision_dataset",
    "load_exported_dataset",
    "plan_vision_cache",
    "validate_study_dataset",
]
