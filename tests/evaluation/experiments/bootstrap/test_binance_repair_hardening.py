"""Falsification contracts for persisted Binance Vision repair evidence."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_binance_repair import (
    _config,
    _freeze,
    _open_ms,
)
from trade_rl.evaluation.experiments.bootstrap.binance import (
    _inspect_frozen_binance_source,
)
from trade_rl.integrations.binance import BinanceTransportError, vision_cache_path


def _resolution(root: Path) -> dict[str, object]:
    payload = json.loads((root / "vision-resolution.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _repair_url(root: Path) -> str:
    payload = _resolution(root)
    repairs = payload["repairs"]
    assert isinstance(repairs, list) and len(repairs) == 1
    repair = repairs[0]
    assert isinstance(repair, dict)
    urls = repair["daily_urls"]
    assert isinstance(urls, list) and len(urls) >= 1
    return str(urls[0])


def test_resolution_tamper_is_recomputed_from_primary_evidence(tmp_path: Path) -> None:
    missing = _open_ms(9, 4)
    root, _live, _frozen = _freeze(tmp_path, missing_primary_opens=(missing,))
    path = root / "vision-resolution.json"
    payload = _resolution(root)
    repairs = payload["repairs"]
    assert isinstance(repairs, list)
    repair = repairs[0]
    assert isinstance(repair, dict)
    repair["missing_open_ms"] = [missing + 60 * 60 * 1_000]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution"):
        _inspect_frozen_binance_source(_config(), root)


def test_repair_url_order_tamper_is_rejected(tmp_path: Path) -> None:
    first = _open_ms(8, 2)
    second = _open_ms(18, 7)
    root, _live, _frozen = _freeze(
        tmp_path,
        missing_primary_opens=(first, second),
    )
    path = root / "vision-resolution.json"
    payload = _resolution(root)
    repairs = payload["repairs"]
    assert isinstance(repairs, list)
    repair = repairs[0]
    assert isinstance(repair, dict)
    urls = repair["daily_urls"]
    assert isinstance(urls, list) and len(urls) == 2
    repair["daily_urls"] = list(reversed(urls))
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution"):
        _inspect_frozen_binance_source(_config(), root)


def test_repair_sidecar_digest_tamper_is_rejected(tmp_path: Path) -> None:
    root, _live, _frozen = _freeze(
        tmp_path,
        missing_primary_opens=(_open_ms(11, 6),),
    )
    cache = vision_cache_path(root / "vision-cache", _repair_url(root))
    sidecar = cache.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(BinanceTransportError, match="digest|mismatch"):
        _inspect_frozen_binance_source(_config(), root)


def test_offline_reinspection_and_repair_load_do_not_use_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _live, _frozen = _freeze(
        tmp_path,
        missing_primary_opens=(_open_ms(14, 9),),
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("offline inspection used network"),
    )

    frozen = _inspect_frozen_binance_source(_config(), root)
    config = _config()
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


def test_offline_inspection_rejects_self_consistent_repair_overlap_conflict(
    tmp_path: Path,
) -> None:
    missing = _open_ms(21, 8)
    overlap = _open_ms(21, 9)
    root, live, _frozen = _freeze(tmp_path, missing_primary_opens=(missing,))
    repair_url = _repair_url(root)
    live.conflicting_daily_open = overlap
    payload = live._payload(repair_url)
    cache = vision_cache_path(root / "vision-cache", repair_url)
    cache.write_bytes(payload)
    sidecar = cache.with_suffix(".json")
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    evidence["sha256"] = hashlib.sha256(payload).hexdigest()
    evidence["size_bytes"] = len(payload)
    sidecar.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="conflict|overlap"):
        _inspect_frozen_binance_source(_config(), root)
