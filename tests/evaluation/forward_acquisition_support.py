"""Offline fixtures for the FS1 acquisition tests: a miniature FS1 span, synthetic
Binance Vision archives, a fake transport and a verified miniature "development" Dataset.

Everything is synthetic and local: no request leaves the process, no real data exists.
The miniature registry keeps the real FS1 structure (a development part that is reused
read-only and a fetched part ending at the scoring end) on a three-month span.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.artifacts.publication import (
    load_published_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.evaluation.forward_acquisition import acquire, plan
from trade_rl.evaluation.forward_study.registry import (
    FS1_BUILD_POLICY,
    FS1_REGISTRY,
    Fs1Registry,
    payload_digest,
)
from trade_rl.evaluation.forward_study.seal import CodeIdentity
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
    FrozenBinanceExchangeInfoTransport,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.dataset import BinanceGapPolicy
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
)

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-424242424"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
HOUR_MS = 3_600_000
# Mini span [2026-01-01, 2026-04-01); development part is January+February.
MINI_START = datetime(2026, 1, 1, tzinfo=UTC)
MINI_DEV_END = datetime(2026, 3, 1, tzinfo=UTC)
MINI_END = datetime(2026, 4, 1, tzinfo=UTC)
# scoring end 2026-04-01T00:00Z plus the 24 h settlement margin
AFTER_SETTLEMENT = datetime(2026, 4, 2, 6, 0, tzinfo=UTC)
CODE_SHA = "c" * 40
EXCHANGE_JSON = json.dumps(
    {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "onboardDate": 1_500_000_000_000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
            for symbol in SYMBOLS
        ]
    },
    sort_keys=True,
).encode()


def mini_registry() -> Fs1Registry:
    policy = {
        **FS1_BUILD_POLICY,
        "symbols": list(SYMBOLS),
        "start_time": MINI_START.isoformat(),
        "end_time": MINI_END.isoformat(),
    }
    return replace(
        FS1_REGISTRY,
        symbols=SYMBOLS,
        dataset_first_row_latest="2026-01-02T00:00:00",
        development_data_end="2026-03-01T00:00:00",
        warmup_only_start="2026-03-01T00:00:00",
        scoring_start="2026-03-16T00:00:00",
        scoring_end="2026-04-01T00:00:00",
        causality_cut="2026-03-20T00:00:00",
        build_policy=policy,
    )


def mini_plan(registry: Fs1Registry) -> plan.AcquisitionPlan:
    params, problems = plan.build_params_or_problems(
        registry,
        implemented_policy_digest=payload_digest(dict(registry.build_policy)),
    )
    assert params is not None, problems
    return plan.plan_acquisition(
        params,
        development_end=MINI_DEV_END,
        scoring_start=datetime(2026, 3, 16, tzinfo=UTC),
    )


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _kline_row(open_ms: int, close: float, index: int) -> str:
    fields = [
        open_ms,
        close * 0.999,
        close * 1.01,
        close * 0.99,
        close,
        10.0,
        open_ms + HOUR_MS - 1,
        1_000.0 + index,
        5,
        1,
        1,
        0,
    ]
    return ",".join(str(value) for value in fields)


def make_archives(
    item_plan: plan.AcquisitionPlan,
    *,
    missing_hours: dict[str, set[datetime]] | None = None,
    missing_everywhere: Iterable[datetime] = (),
) -> dict[str, bytes]:
    """Synthetic Vision archive bytes for every planned archive, keyed by URL."""

    missing_hours = missing_hours or {}
    everywhere = set(missing_everywhere)
    archives: dict[str, bytes] = {}
    prices: dict[str, float] = {}
    for item in item_plan.archives:
        month_start = datetime.strptime(item.period, "%Y-%m").replace(tzinfo=UTC)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        name = item.url.rsplit("/", 1)[1].removesuffix(".zip") + ".csv"
        if item.kind == plan.KIND_KLINES:
            rng = np.random.default_rng(
                int(hashlib.sha256(item.symbol.encode()).hexdigest()[:8], 16)
            )
            price = prices.get(item.symbol, 100.0)
            rows: list[str] = []
            hour = month_start
            index = 0
            while hour < next_month:
                price *= float(np.exp(rng.normal(0.0, 0.005)))
                skipped = hour in everywhere or hour in missing_hours.get(
                    item.symbol, set()
                )
                if not skipped:
                    rows.append(_kline_row(int(hour.timestamp() * 1000), price, index))
                hour += timedelta(hours=1)
                index += 1
            prices[item.symbol] = price
            archives[item.url] = _zip(name, "\n".join(rows) + "\n")
        else:
            lines = ["calc_time,funding_interval_hours,last_funding_rate"]
            day = month_start
            while day < next_month:
                for hours in (0, 8, 16):
                    when = day + timedelta(hours=hours, milliseconds=1)
                    lines.append(f"{int(when.timestamp() * 1000)},8,0.0001")
                day += timedelta(days=1)
            archives[item.url] = _zip(name, "\n".join(lines) + "\n")
    return archives


def checksum_text(url: str, payload: bytes) -> bytes:
    filename = url.rsplit("/", 1)[1]
    return f"{hashlib.sha256(payload).hexdigest()}  {filename}\n".encode()


class FakeTransport(BinancePublicTransport):
    """Serves fixture bytes and records every URL; the same cache layout as the real one."""

    def __init__(
        self,
        cache_root: Path,
        served: dict[str, bytes],
        requests: list[str],
        retrieved_at: datetime,
    ) -> None:
        super().__init__(cache_root=cache_root, allow_network=True)
        self.served = served
        self.requests = requests
        self.retrieved_at = retrieved_at

    def _request_bytes(
        self, url: str, *, max_response_bytes: int | None = None
    ) -> bytes:
        cache_path = self._vision_cache_path(url)
        if cache_path is not None and cache_path.is_file():
            return self._validated_cached_vision_payload(url, cache_path)
        self.requests.append(url)
        if url not in self.served:
            raise BinanceTransportError(f"fixture has no such object (404): {url}")
        payload = self.served[url]
        if cache_path is not None:
            self._write_vision_cache(
                url=url,
                cache_path=cache_path,
                payload=payload,
                etag=None,
                last_modified=None,
            )
        return payload

    def load_exchange_information_snapshot(  # type: ignore[override]
        self, *, market: Any, mode: Any = "rest"
    ) -> Any:
        return super().load_exchange_information_snapshot(
            market=market, mode=mode, clock=lambda: self.retrieved_at
        )


@dataclass
class Scenario:
    """A complete miniature world: registry, archives, development inputs, environment."""

    registry: Fs1Registry
    item_plan: plan.AcquisitionPlan
    archives: dict[str, bytes]
    dev_archive_root: Path
    dev_source_evidence: Path
    dev_dataset: Path
    dev_reference: acquire.DevReference
    served: dict[str, bytes]
    requests: list[str]
    clock_calls: list[datetime]

    def environment(
        self,
        *,
        registry: Fs1Registry | None = None,
        implemented_policy_digest: str | None = None,
        dev_reference: acquire.DevReference | None = None,
        now: datetime = AFTER_SETTLEMENT,
        code_identity: CodeIdentity | None = None,
        transport_factory: Callable[[Path], Any] | None = None,
    ) -> acquire._Environment:
        chosen = registry or self.registry
        ticks = iter(range(10_000))

        def clock() -> datetime:
            value = now + timedelta(seconds=next(ticks))
            self.clock_calls.append(value)
            return value

        return acquire._Environment(
            registry=chosen,
            frozen_params_digest=chosen.digest(),
            implemented_policy_digest=implemented_policy_digest
            or payload_digest(dict(self.registry.build_policy)),
            dev_reference=dev_reference or self.dev_reference,
            clock=clock,
            code_identity=lambda: code_identity or CodeIdentity(CODE_SHA, True),
            network_transport=transport_factory
            or (
                lambda cache_root: FakeTransport(
                    cache_root, self.served, self.requests, AFTER_SETTLEMENT
                )
            ),
        )

    def run(self, root: Path, **overrides: Any) -> acquire.AcquisitionResult:
        env_keys = {
            "registry",
            "implemented_policy_digest",
            "dev_reference",
            "now",
            "code_identity",
            "transport_factory",
        }
        env_args = {k: overrides.pop(k) for k in list(overrides) if k in env_keys}
        arguments: dict[str, Any] = {
            "authorization": AUTH,
            "output_root": root,
            "allow_network": True,
            "dev_archive_root": self.dev_archive_root,
            "dev_source_evidence": self.dev_source_evidence,
            "dev_dataset": self.dev_dataset,
            "command_line": "python -m trade_rl.evaluation.forward_acquisition acquire fixture",
        }
        arguments.update(overrides)
        return acquire._run_acquire(self.environment(**env_args), **arguments)


def _write_dev_files(
    item_plan: plan.AcquisitionPlan, archives: dict[str, bytes], root: Path
) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    for item in item_plan.reused:
        path = root / "archives" / item.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archives[item.url])
        roster.append(
            {
                "url": item.url,
                "sha256": hashlib.sha256(archives[item.url]).hexdigest(),
                "size_bytes": len(archives[item.url]),
            }
        )
    return roster


def _build_dev_dataset(
    item_plan: plan.AcquisitionPlan,
    archives: dict[str, bytes],
    roster: list[dict[str, Any]],
    root: Path,
) -> acquire.DevReference:
    """A real build of the development span, published like the development Dataset."""

    cache_root = root / "dev-cache"
    exchange_dir = root / "dev-exchange"
    exchange_dir.mkdir(parents=True)
    (exchange_dir / "exchange-info.raw.json").write_bytes(EXCHANGE_JSON)
    (exchange_dir / "manifest.json").write_text(
        json.dumps(
            {
                "market": "usds-m",
                "raw_payload_sha256": hashlib.sha256(EXCHANGE_JSON).hexdigest(),
                "retrieved_at": "2026-03-01T00:00:00+00:00",
                "schema_version": "frozen_metadata_cache_v1",
                "source_uri": "https://fapi.binance.com/fapi/v1/exchangeInfo",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    writer = BinancePublicTransport(cache_root=cache_root, allow_network=False)
    for item in item_plan.reused:
        path = writer._vision_cache_path(item.url)
        assert path is not None
        writer._write_vision_cache(
            url=item.url,
            cache_path=path,
            payload=archives[item.url],
            etag=None,
            last_modified=None,
        )
    transport = BaseIntervalAggregatingTransport(
        BinancePublicTransport(cache_root=cache_root, allow_network=False),
        exchange_information=FrozenBinanceExchangeInfoTransport(exchange_dir),
        base_interval="1h",
    )
    built = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=SYMBOLS,
        interval="1h",
        start_time=MINI_START,
        end_time=MINI_DEV_END,
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        feature_timeframes=("4h", "1d"),
        metadata_evidence={
            "schema_version": "mini_dev_source_evidence",
            "raw_source_roster_digest": content_digest(roster),
            "raw_source_count": len(roster),
        },
        execution_economics=ExecutionEconomicsProfile(
            name="bot_overlay_only_assumptions_v1"
        ),
        gap_policy=BinanceGapPolicy.NON_TRADABLE_ROWS,
    )
    published = publish_market_dataset_artifact(root / "dev-dataset", built.dataset)
    _, loaded = load_published_market_dataset_artifact(root / "dev-dataset")
    assert loaded.identity_verified
    return acquire.DevReference(loaded.dataset_id, published.artifact_digest)


def make_scenario(
    root: Path,
    *,
    missing_hours: dict[str, set[datetime]] | None = None,
    missing_everywhere: Iterable[datetime] = (),
    wrong_checksum_for: str | None = None,
    omit_from_server: Iterable[str] = (),
) -> Scenario:
    """Build the fixture world under ``root``.

    ``missing_*`` remove kline hours from the archives of the FETCHED month only (the
    development archives are fixed by the development identity);
    ``wrong_checksum_for`` serves a wrong ``.CHECKSUM`` for one planned URL;
    ``omit_from_server`` makes URLs answer like a 404.
    """

    registry = mini_registry()
    item_plan = mini_plan(registry)
    archives = make_archives(
        item_plan,
        missing_hours=missing_hours,
        missing_everywhere=missing_everywhere,
    )
    roster = _write_dev_files(item_plan, archives, root)
    evidence_path = root / "source-evidence.json"
    evidence_path.write_text(
        json.dumps({"evidence": {}, "roster": roster}, sort_keys=True),
        encoding="utf-8",
    )
    dev_reference = _build_dev_dataset(item_plan, archives, roster, root)
    served: dict[str, bytes] = {plan.EXCHANGE_INFO_URL: EXCHANGE_JSON}
    for item in item_plan.fetched:
        served[item.url] = archives[item.url]
        served[item.checksum_url] = checksum_text(item.url, archives[item.url])
    if wrong_checksum_for is not None:
        served[wrong_checksum_for + plan.CHECKSUM_SUFFIX] = checksum_text(
            wrong_checksum_for, b"not the archive"
        )
    for url in omit_from_server:
        served.pop(url)
    return Scenario(
        registry=registry,
        item_plan=item_plan,
        archives=archives,
        dev_archive_root=root / "archives",
        dev_source_evidence=evidence_path,
        dev_dataset=root / "dev-dataset",
        dev_reference=dev_reference,
        served=served,
        requests=[],
        clock_calls=[],
    )
