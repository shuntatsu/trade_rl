from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.bootstrap.binance import _freeze_binance_source
from trade_rl.evaluation.experiments.bootstrap.config import CanonicalM2BootstrapConfig
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.runs import CandidateRunConfig
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinanceTransportMode,
    plan_binance_vision_cache,
    vision_cache_path,
    vision_kline_url,
)

_HOUR_MS = 60 * 60 * 1_000


def _config() -> CanonicalM2BootstrapConfig:
    return CanonicalM2BootstrapConfig(
        research_question="repair arbitrary incomplete monthly Vision evidence",
        market=BinanceMarket.USDS_M,
        symbols=("TESTUSDT",),
        base_timeframe="1h",
        feature_timeframes=(),
        data_start=datetime(2024, 1, 1, tzinfo=UTC),
        data_stop_exclusive=datetime(2024, 2, 1, tzinfo=UTC),
        baseline=CandidateRunConfig(
            signal_name="1h__log_return_24bar",
            feature_names=("1h__log_return_24bar",),
            fit_symbol_names=("TESTUSDT",),
            fit_cutoff="2024-01-15T00:00:00",
            evaluation_start="2024-01-15T00:00:00",
            evaluation_stop_exclusive="2024-02-01T00:00:00",
            rule_entry_threshold=0.01,
            rule_exit_threshold=0.005,
            forecast_entry_threshold=0.01,
            forecast_exit_threshold=0.005,
            ppo_total_timesteps=100,
            ppo_seed=0,
            gross_budget=0.5,
            initial_capital=100_000.0,
        ),
        ppo_seeds=(0, 1),
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=4,
        n_bootstrap=100,
        bootstrap_seed=7,
    )


def _zip_csv(name: str, rows: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, "\n".join(rows) + "\n")
    return buffer.getvalue()


def _kline_row(open_ms: int, *, close: str = "100.5") -> str:
    return f"{open_ms},100,101,99,{close},1,{open_ms + _HOUR_MS - 1},1000"


def _month_start(url: str) -> datetime:
    match = re.search(r"-(20\d{2})-(\d{2})\.zip$", url)
    assert match is not None, url
    return datetime(int(match.group(1)), int(match.group(2)), 1, tzinfo=UTC)


def _day_start(url: str) -> datetime:
    match = re.search(r"-(20\d{2})-(\d{2})-(\d{2})\.zip$", url)
    assert match is not None, url
    return datetime(
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        tzinfo=UTC,
    )


def _snapshot() -> BinanceExchangeInfoSnapshot:
    raw = json.dumps(
        {
            "symbols": [
                {
                    "symbol": "TESTUSDT",
                    "status": "TRADING",
                    "onboardDate": 0,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    ],
                }
            ]
        },
        separators=(",", ":"),
    ).encode()
    return BinanceExchangeInfoSnapshot(
        payload=json.loads(raw),
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 9, 11, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _RepairFixtureTransport:
    def __init__(
        self,
        cache_root: Path,
        *,
        missing_primary_opens: tuple[int, ...] = (),
        missing_daily_opens: tuple[int, ...] = (),
        conflicting_daily_open: int | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.missing_primary_opens = frozenset(missing_primary_opens)
        self.missing_daily_opens = frozenset(missing_daily_opens)
        self.conflicting_daily_open = conflicting_daily_open
        self.download_calls: list[str] = []

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> BinanceExchangeInfoSnapshot:
        assert BinanceMarket(market) is BinanceMarket.USDS_M
        assert BinanceTransportMode(mode) is BinanceTransportMode.REST
        return _snapshot()

    def _request_bytes(self, url: str) -> bytes:
        self.download_calls.append(url)
        payload = self._payload(url)
        path = vision_cache_path(self.cache_root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "acquired_at": "2026-09-11T00:00:00+00:00",
                    "downloader": "test",
                    "etag": None,
                    "last_modified": None,
                    "schema_version": "binance_vision_raw_cache_v1",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "url": url,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return payload

    def _payload(self, url: str) -> bytes:
        if "fundingRate" in url:
            start_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000)
            return _zip_csv(
                "funding.csv",
                [
                    "calc_time,last_funding_rate",
                    f"{start_ms + 8 * _HOUR_MS},0.0001",
                ],
            )
        if "/monthly/klines/" in url:
            start = _month_start(url)
            stop = datetime(2024, 2, 1, tzinfo=UTC)
            rows: list[str] = []
            cursor = start
            while cursor < stop:
                open_ms = int(cursor.timestamp() * 1_000)
                if open_ms not in self.missing_primary_opens:
                    rows.append(_kline_row(open_ms))
                cursor += timedelta(hours=1)
            return _zip_csv("monthly.csv", rows)
        if "/daily/klines/" in url:
            start = _day_start(url)
            rows = []
            for hour in range(24):
                open_ms = int((start + timedelta(hours=hour)).timestamp() * 1_000)
                if open_ms in self.missing_daily_opens:
                    continue
                close = "999.5" if open_ms == self.conflicting_daily_open else "100.5"
                rows.append(_kline_row(open_ms, close=close))
            return _zip_csv("daily.csv", rows)
        raise AssertionError(f"unexpected URL: {url}")


def _open_ms(day: int, hour: int) -> int:
    return int(datetime(2024, 1, day, hour, tzinfo=UTC).timestamp() * 1_000)


def _freeze(
    tmp_path: Path,
    **transport_kwargs: object,
):
    root = tmp_path / "source"
    live = _RepairFixtureTransport(root / "vision-cache", **transport_kwargs)
    frozen = _freeze_binance_source(_config(), root, live_transport=live)
    return root, live, frozen


def test_freeze_repairs_arbitrary_primary_gap_with_explicit_daily_evidence(
    tmp_path: Path,
) -> None:
    missing = _open_ms(10, 5)
    root, live, frozen = _freeze(tmp_path, missing_primary_opens=(missing,))
    config = _config()
    primary = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )
    repair_url = vision_kline_url(
        config.market,
        "TESTUSDT",
        "1h",
        datetime(2024, 1, 10, tzinfo=UTC),
    )

    plan = json.loads((root / "vision-plan.json").read_text(encoding="utf-8"))
    resolution = json.loads(
        (root / "vision-resolution.json").read_text(encoding="utf-8")
    )
    assert tuple(plan["urls"]) == primary.urls
    assert resolution["schema_version"] == "canonical_m2_vision_resolution_v1"
    assert resolution["repairs"] == [
        {
            "daily_urls": [repair_url],
            "missing_open_ms": [missing],
            "symbol": "TESTUSDT",
            "timeframe": "1h",
        }
    ]
    assert live.download_calls == [*primary.urls, repair_url]

    rows, source = frozen.composite_transport.load_klines(
        market=config.market,
        symbol="TESTUSDT",
        interval="1h",
        start_ms=int(config.data_start.timestamp() * 1_000),
        end_ms=int(config.data_stop_exclusive.timestamp() * 1_000),
        mode="vision",
    )
    assert source == "vision"
    assert len(rows) == 31 * 24
    assert [int(row[0]) for row in rows] == [
        int((config.data_start + timedelta(hours=index)).timestamp() * 1_000)
        for index in range(31 * 24)
    ]


def test_freeze_complete_primary_clock_does_not_request_daily_repair(
    tmp_path: Path,
) -> None:
    root, live, _frozen = _freeze(tmp_path)
    config = _config()
    primary = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )

    resolution = json.loads(
        (root / "vision-resolution.json").read_text(encoding="utf-8")
    )
    assert resolution["repairs"] == []
    assert live.download_calls == list(primary.urls)


def test_freeze_rejects_daily_repair_that_still_misses_required_open(
    tmp_path: Path,
) -> None:
    first = _open_ms(12, 3)
    second = _open_ms(12, 4)

    with pytest.raises(ValueError, match="complete|missing|repair"):
        _freeze(
            tmp_path,
            missing_primary_opens=(first, second),
            missing_daily_opens=(second,),
        )


def test_freeze_rejects_semantic_conflict_in_daily_overlap(tmp_path: Path) -> None:
    missing = _open_ms(20, 8)
    overlapping = _open_ms(20, 9)

    with pytest.raises(ValueError, match="conflict|overlap"):
        _freeze(
            tmp_path,
            missing_primary_opens=(missing,),
            conflicting_daily_open=overlapping,
        )
