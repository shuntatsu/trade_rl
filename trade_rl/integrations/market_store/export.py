"""Build an existing :class:`~trade_rl.data.MarketDataset` from stored market data.

Nothing here re-implements dataset assembly. The stored rows are served through the
same transport interface the Binance Vision file path uses, so the unchanged
:func:`~trade_rl.integrations.binance.build_binance_market_dataset` (gap policy,
higher-timeframe aggregation, features, identity) builds the Dataset:

* klines come back as the Vision column layout from exact stored NUMERIC values;
* funding events come back as ``(timestamp_ms, rate)`` with the Vision inclusive range;
* exchange rules come from one stored raw ``exchangeInfo`` snapshot, applied
  statically, exactly like the existing frozen-metadata path.

Determinism: all rows are read first and digested; the digests and the exchange-info
snapshot digest enter the Dataset identity as ``metadata_evidence``. The same store
content and parameters therefore always give the same ``dataset_id``. The default
snapshot is the latest stored one, so pass ``rules_snapshot_sha256`` to pin it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts import content_digest
from trade_rl.data import MarketDataset, publish_market_dataset_artifact
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.integrations.binance import BinanceMarket, build_binance_market_dataset
from trade_rl.integrations.binance.dataset import (
    BinanceDatasetBuildResult,
    BinanceGapPolicy,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)
from trade_rl.integrations.binance.metadata import _mutable_json_object
from trade_rl.integrations.market_store.contracts import (
    FundingRow,
    KlineRow,
    MarketReader,
    StoredExchangeInfo,
    StoreMarket,
    WarmupRun,
    funding_digest,
    klines_digest,
    ms_from_utc,
    utc_from_ms,
)
from trade_rl.integrations.market_store.embargo import (
    EmbargoPolicy,
    SealedWindowEmbargoError,
)

EXPORT_SCHEMA = "market_store_export_v1"
ECONOMICS_PROFILE_NAME = "bot_overlay_only_assumptions_v1"
DEFAULT_FEATURE_TIMEFRAMES = ("4h", "1d")
TRANSPORT_SOURCE = "market_store"
EXCHANGE_INFO_SOURCE = "market_store:exchange-info"


class StoreCoverageError(RuntimeError):
    """The store holds no rows for a requested series (nothing to build from)."""


class StoreMarketTransport:
    """The Vision-transport interface, answered from pre-read store rows."""

    def __init__(
        self,
        *,
        market: StoreMarket,
        interval: str,
        start_ms: int,
        end_ms: int,
        klines: Mapping[str, Sequence[KlineRow]],
        funding: Mapping[str, Sequence[FundingRow]],
        exchange_info: StoredExchangeInfo,
    ) -> None:
        self._market = market
        self._interval = interval
        self._start_ms = start_ms
        self._end_ms = end_ms
        self._klines = {symbol: list(rows) for symbol, rows in klines.items()}
        self._funding = {symbol: list(rows) for symbol, rows in funding.items()}
        self._exchange_info = exchange_info

    def _require_market(self, market: object) -> None:
        if StoreMarket.from_binance(str(getattr(market, "value", market))) is not (
            self._market
        ):
            raise ValueError("the store transport serves exactly one market")

    def load_klines(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        mode: object = None,
    ) -> tuple[list[list[object]], str]:
        self._require_market(market)
        if interval != self._interval:
            raise ValueError(
                f"the store export serves only the {self._interval} base interval"
            )
        if start_ms < self._start_ms or end_ms > self._end_ms:
            raise ValueError("requested range lies outside the exported range")
        rows = self._klines.get(symbol)
        if rows is None:
            raise StoreCoverageError(f"symbol {symbol} was not loaded from the store")
        return (
            [row.vision_row() for row in rows if start_ms <= row.open_time_ms < end_ms],
            TRANSPORT_SOURCE,
        )

    def load_funding_rates(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        mode: object = None,
    ) -> tuple[list[tuple[int, float]], str]:
        self._require_market(market)
        if self._market is StoreMarket.SPOT:
            return [], "spot:no-funding"
        rows = self._funding.get(symbol)
        if rows is None:
            raise StoreCoverageError(f"symbol {symbol} was not loaded from the store")
        return (
            [
                (row.funding_time_ms, float(row.rate))
                for row in rows
                if start_ms <= row.funding_time_ms <= end_ms
            ],
            TRANSPORT_SOURCE,
        )

    def load_exchange_information(
        self, *, market: BinanceMarket | str, mode: object = None
    ) -> tuple[dict[str, object], str]:
        self._require_market(market)
        payload = json.loads(self._exchange_info.raw_payload)
        if not isinstance(payload, dict):
            raise ValueError("stored exchangeInfo payload must be an object")
        return _mutable_json_object(payload), EXCHANGE_INFO_SOURCE


@dataclass(frozen=True, slots=True)
class StoreExport:
    dataset: MarketDataset
    build: BinanceDatasetBuildResult
    evidence: Mapping[str, Any]
    exchange_info_sha256: str

    def receipt(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset.dataset_id,
            "evidence": dict(self.evidence),
            "evidence_digest": content_digest(dict(self.evidence)),
            "exchange_info_sha256": self.exchange_info_sha256,
            "feature_timeframes": list(self.build.feature_timeframes),
            "gap_report": {k: dict(v) for k, v in self.build.gap_report.items()},
            "n_bars": self.dataset.n_bars,
            "schema_version": EXPORT_SCHEMA,
            "sources_used": list(self.build.sources_used),
            "symbols": list(self.dataset.symbols),
        }


def export_evidence(
    *,
    market: StoreMarket,
    symbols: Sequence[str],
    interval: str,
    start_time: datetime,
    end_time: datetime,
    feature_timeframes: Sequence[str],
    gap_policy: BinanceGapPolicy,
    klines: Mapping[str, Sequence[KlineRow]],
    funding: Mapping[str, Sequence[FundingRow]],
    exchange_info: StoredExchangeInfo,
    warmup_runs: Sequence[WarmupRun] = (),
) -> dict[str, Any]:
    """The Dataset-identity provenance: parameters plus digests of every row read.

    ``fs1_warmup`` records whether any exported bar lies in the FS1 warm-up span and,
    if so, the authorizing ingestion runs (id + Issue #810 URL), so such a Dataset can
    never share an identity or receipt with an ordinary one.
    """

    last_open_ms = max(
        (rows[-1].open_time_ms for rows in klines.values() if rows), default=None
    )
    touches = last_open_ms is not None and last_open_ms >= ms_from_utc(
        EmbargoPolicy.enforced().warmup_start
    )
    return {
        "schema_version": EXPORT_SCHEMA,
        "market": market.value,
        "symbols": list(symbols),
        "interval": interval,
        "start_time": start_time.astimezone(UTC).isoformat(),
        "end_time": end_time.astimezone(UTC).isoformat(),
        "feature_timeframes": list(feature_timeframes),
        "gap_policy": gap_policy.value,
        "klines": {
            symbol: {
                "rows": len(klines[symbol]),
                "sha256": klines_digest(klines[symbol]),
            }
            for symbol in symbols
        },
        "funding_rates": {
            symbol: {
                "rows": len(funding.get(symbol, ())),
                "sha256": funding_digest(funding.get(symbol, ())),
            }
            for symbol in symbols
        },
        "fs1_warmup": {
            "touches_span": touches,
            "last_bar_open_time": None
            if last_open_ms is None
            else utc_from_ms(last_open_ms).isoformat(),
            "runs": [run.to_dict() for run in warmup_runs],
        },
        "exchange_information": {
            "raw_payload_sha256": exchange_info.raw_payload_sha256,
            "source_uri": exchange_info.source_uri,
            "retrieved_at": exchange_info.retrieved_at.astimezone(UTC).isoformat(),
            "rules_assumption": (
                "stored snapshot applied statically; not historical point-in-time rules"
            ),
        },
    }


def export_market_dataset(
    reader: MarketReader,
    *,
    market: StoreMarket,
    symbols: Sequence[str],
    interval: str,
    start_time: datetime,
    end_time: datetime,
    feature_timeframes: Sequence[str] = DEFAULT_FEATURE_TIMEFRAMES,
    gap_policy: BinanceGapPolicy = BinanceGapPolicy.NON_TRADABLE_ROWS,
    rules_snapshot_sha256: str | None = None,
    embargo: EmbargoPolicy | None = None,
    economics_profile_name: str = ECONOMICS_PROFILE_NAME,
) -> StoreExport:
    """Build a Dataset from the store with the existing builder; refuse closed FS1 spans."""

    policy = EmbargoPolicy.enforced() if embargo is None else embargo
    end_utc = end_time.astimezone(UTC)
    policy.check_range_end(end_utc, what="export")
    state = reader.embargo_state()
    if end_utc > state.sealed_from:
        raise SealedWindowEmbargoError(
            f"export end {end_utc.isoformat()} is past the store embargo boundary "
            f"{state.sealed_from.isoformat()}; refusing unconditionally"
        )
    resolved_symbols = tuple(symbols)
    start_ms = ms_from_utc(start_time)
    end_ms = ms_from_utc(end_utc)

    klines: dict[str, list[KlineRow]] = {}
    funding: dict[str, list[FundingRow]] = {}
    for symbol in resolved_symbols:
        rows = reader.read_klines(market, symbol, interval, start_ms, end_ms)
        if not rows:
            raise StoreCoverageError(
                f"the store holds no {market.value} {symbol} {interval} klines in "
                f"[{start_time.isoformat()}, {end_utc.isoformat()})"
            )
        klines[symbol] = rows
        if market is StoreMarket.USDS_M:
            funding[symbol] = reader.read_funding(market, symbol, start_ms, end_ms)
    exchange_info = reader.load_exchange_info(market, rules_snapshot_sha256)

    last_open = max(rows[-1].open_time_ms for rows in klines.values())
    warmup_runs: list[WarmupRun] = []
    if last_open >= ms_from_utc(policy.warmup_start):
        warmup_runs = reader.warmup_provenance(
            market, resolved_symbols, interval, start_ms, end_ms
        )
    evidence = export_evidence(
        market=market,
        symbols=resolved_symbols,
        interval=interval,
        start_time=start_time,
        end_time=end_utc,
        feature_timeframes=feature_timeframes,
        gap_policy=gap_policy,
        klines=klines,
        funding=funding,
        exchange_info=exchange_info,
        warmup_runs=warmup_runs,
    )
    store_transport = StoreMarketTransport(
        market=market,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        klines=klines,
        funding=funding,
        exchange_info=exchange_info,
    )
    transport = BaseIntervalAggregatingTransport(
        store_transport,
        exchange_information=store_transport,
        base_interval=interval,
    )
    build = build_binance_market_dataset(
        market=market.binance,
        symbols=resolved_symbols,
        interval=interval,
        start_time=start_time,
        end_time=end_utc,
        transport_mode="vision",
        transport=transport,
        feature_timeframes=tuple(feature_timeframes),
        metadata_evidence=evidence,
        execution_economics=ExecutionEconomicsProfile(name=economics_profile_name),
        gap_policy=gap_policy,
    )
    return StoreExport(
        dataset=build.dataset,
        build=build,
        evidence=evidence,
        exchange_info_sha256=exchange_info.raw_payload_sha256,
    )


def publish_export(export: StoreExport, output_dir: Path) -> dict[str, object]:
    """Publish the Dataset artifact and a receipt into a new empty ``output_dir``."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    published = publish_market_dataset_artifact(output_dir / "dataset", export.dataset)
    receipt = export.receipt() | {"artifact_digest": published.artifact_digest}
    (output_dir / "export-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


__all__ = [
    "DEFAULT_FEATURE_TIMEFRAMES",
    "ECONOMICS_PROFILE_NAME",
    "EXPORT_SCHEMA",
    "StoreCoverageError",
    "StoreExport",
    "StoreMarketTransport",
    "export_evidence",
    "export_market_dataset",
    "publish_export",
]
