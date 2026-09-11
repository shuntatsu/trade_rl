from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.binance import (
    _freeze_binance_source,
    _inspect_frozen_binance_source,
)
from trade_rl.evaluation.experiments.bootstrap.config import CanonicalM2BootstrapConfig
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.runs.config import CandidateRunConfig
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    binance_interval_milliseconds,
    plan_binance_vision_cache,
    vision_cache_path,
)


def _config() -> CanonicalM2BootstrapConfig:
    return CanonicalM2BootstrapConfig(
        research_question="freeze exact Binance source evidence",
        market=BinanceMarket.USDS_M,
        symbols=("BTCUSDT", "ETHUSDT"),
        base_timeframe="1h",
        feature_timeframes=("4h",),
        data_start=datetime(2024, 1, 1, tzinfo=UTC),
        data_stop_exclusive=datetime(2024, 3, 1, tzinfo=UTC),
        baseline=CandidateRunConfig(
            signal_name="1h__log_return_24bar",
            feature_names=("1h__log_return_24bar",),
            fit_symbol_names=("BTCUSDT", "ETHUSDT"),
            fit_cutoff="2024-02-01T00:00:00",
            evaluation_start="2024-02-01T00:00:00",
            evaluation_stop_exclusive="2024-03-01T00:00:00",
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


def _zip_csv(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _month_start_ms(url: str) -> int:
    match = re.search(r"-(20\d{2})-(\d{2})\.zip$", url)
    assert match is not None, url
    return int(
        datetime(int(match.group(1)), int(match.group(2)), 1, tzinfo=UTC).timestamp()
        * 1_000
    )


def _payload_for_url(url: str) -> bytes:
    start_ms = _month_start_ms(url)
    if "fundingRate" in url:
        return _zip_csv(
            "funding.csv",
            f"calc_time,last_funding_rate\n{start_ms + 8 * 60 * 60 * 1000},0.0001\n",
        )
    interval_match = re.search(r"/(15m|30m|1h|2h|4h|6h|8h|12h|1d)/", url)
    assert interval_match is not None, url
    step = binance_interval_milliseconds(interval_match.group(1))
    start = datetime.fromtimestamp(start_ms / 1_000, tz=UTC)
    if start.month == 12:
        stop = start.replace(year=start.year + 1, month=1)
    else:
        stop = start.replace(month=start.month + 1)
    stop_ms = int(stop.timestamp() * 1_000)
    rows = []
    for open_ms in range(start_ms, stop_ms, step):
        rows.append(f"{open_ms},100,101,99,100.5,1,{open_ms + step - 1},1000")
    return _zip_csv("klines.csv", "\n".join(rows) + "\n")


def _snapshot() -> BinanceExchangeInfoSnapshot:
    raw = json.dumps(
        {
            "symbols": [
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "onboardDate": 0,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    ],
                }
                for symbol in ("BTCUSDT", "ETHUSDT")
            ]
        },
        separators=(",", ":"),
    ).encode()
    return BinanceExchangeInfoSnapshot(
        payload=json.loads(raw),
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _FakeLiveTransport:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.snapshot_calls = 0
        self.download_calls: list[str] = []

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.REST,
    ) -> BinanceExchangeInfoSnapshot:
        assert BinanceMarket(market) is BinanceMarket.USDS_M
        assert BinanceTransportMode(mode) is BinanceTransportMode.REST
        self.snapshot_calls += 1
        return _snapshot()

    def _request_bytes(self, url: str) -> bytes:
        self.download_calls.append(url)
        payload = _payload_for_url(url)
        path = vision_cache_path(self.cache_root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "acquired_at": "2026-09-01T00:00:00+00:00",
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


def _freeze(tmp_path: Path):
    root = tmp_path / "source"
    live = _FakeLiveTransport(root / "vision-cache")
    frozen = _freeze_binance_source(_config(), root, live_transport=live)
    return root, live, frozen


def test_freeze_binds_exact_plan_raw_roster_and_metadata(tmp_path: Path) -> None:
    config = _config()
    root, live, frozen = _freeze(tmp_path)
    expected = plan_binance_vision_cache(
        market=config.market,
        symbols=config.symbols,
        intervals=(config.base_timeframe, *config.feature_timeframes),
        start_time=config.data_start,
        end_time=config.data_stop_exclusive,
    )

    assert live.snapshot_calls == 1
    assert tuple(live.download_calls) == expected.urls
    plan_payload = json.loads((root / "vision-plan.json").read_text(encoding="utf-8"))
    assert tuple(plan_payload["urls"]) == expected.urls
    assert frozen.vision_plan_digest == content_digest(plan_payload)
    assert tuple(item["url"] for item in frozen.raw_source_roster) == expected.urls
    assert frozen.raw_source_roster_digest == content_digest(
        list(frozen.raw_source_roster)
    )
    for item in frozen.raw_source_roster:
        assert set(item) == {"url", "sha256", "size_bytes"}
        assert len(item["sha256"]) == 64
        assert item["size_bytes"] > 0
    assert frozen.metadata_evidence == {
        "schema_version": "frozen_binance_exchange_info_evidence_v1",
        "market": "usds-m",
        "source_uri": "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "raw_payload_sha256": _snapshot().raw_payload_sha256,
        "retrieved_at": "2026-09-01T00:00:00+00:00",
    }


def test_frozen_composite_is_network_cut_and_cache_miss_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _live, frozen = _freeze(tmp_path)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("frozen source attempted HTTP"),
    )
    start_ms = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    end_ms = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1_000)

    rows, kline_source = frozen.composite_transport.load_klines(
        market="usds-m",
        symbol="BTCUSDT",
        interval="1h",
        start_ms=start_ms,
        end_ms=end_ms,
        mode="vision",
    )
    funding, funding_source = frozen.composite_transport.load_funding_rates(
        market="usds-m",
        symbol="BTCUSDT",
        start_ms=start_ms,
        end_ms=end_ms,
        mode="vision",
    )
    metadata, metadata_source = frozen.composite_transport.load_exchange_information(
        market="usds-m"
    )
    assert rows
    assert funding
    assert kline_source == "vision"
    assert funding_source == "vision"
    assert metadata_source == "frozen:exchange-info"
    assert metadata["symbols"][0]["symbol"] == "BTCUSDT"

    first = frozen.raw_source_roster[0]["url"]
    assert isinstance(first, str)
    vision_cache_path(root / "vision-cache", first).unlink()
    with pytest.raises(BinanceTransportError, match="network access is disabled"):
        frozen.cache_transport._request_bytes(first)


def test_inspection_rejects_raw_archive_tamper(tmp_path: Path) -> None:
    root, _live, frozen = _freeze(tmp_path)
    url = frozen.raw_source_roster[0]["url"]
    assert isinstance(url, str)
    path = vision_cache_path(root / "vision-cache", url)
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises((BinanceTransportError, ValueError, FileNotFoundError)):
        _inspect_frozen_binance_source(_config(), root)


def test_inspection_rejects_sidecar_tamper(tmp_path: Path) -> None:
    root, _live, frozen = _freeze(tmp_path)
    url = frozen.raw_source_roster[0]["url"]
    assert isinstance(url, str)
    path = vision_cache_path(root / "vision-cache", url).with_suffix(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((BinanceTransportError, ValueError, FileNotFoundError)):
        _inspect_frozen_binance_source(_config(), root)


def test_inspection_rejects_vision_plan_order_tamper(tmp_path: Path) -> None:
    root, _live, _frozen = _freeze(tmp_path)
    plan_path = root / "vision-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["urls"] = list(reversed(payload["urls"]))
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Vision plan"):
        _inspect_frozen_binance_source(_config(), root)


def test_inspection_rejects_exchange_info_tamper(tmp_path: Path) -> None:
    root, _live, _frozen = _freeze(tmp_path)
    raw = root / "exchange-info" / "exchange-info.raw.json"
    raw.write_bytes(raw.read_bytes() + b" ")

    with pytest.raises((RuntimeError, ValueError), match="digest|metadata"):
        _inspect_frozen_binance_source(_config(), root)
