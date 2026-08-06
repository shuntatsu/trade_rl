from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "binance-multitimeframe"
        / "sync_market_data.py"
    )
    spec = importlib.util.spec_from_file_location("sync_market_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_maintained_plan_uses_single_btc_pipeline_identity() -> None:
    module = _load_module()

    plan = module.build_maintained_plan()

    assert plan.symbols == ("BTCUSDT",)
    assert plan.symbols == tuple(module.pipeline._SYMBOLS)
    assert plan.intervals == tuple(module.pipeline._NATIVE_TIMEFRAMES)
    assert plan.start_time == datetime.fromisoformat(
        module.pipeline._START.replace("Z", "+00:00")
    ).astimezone(UTC)
    assert plan.end_time == datetime.fromisoformat(
        module.pipeline._END.replace("Z", "+00:00")
    ).astimezone(UTC)
    assert plan.urls
    assert all("BTCUSDT" in url for url in plan.urls)
    assert all("ETHUSDT" not in url and "BNBUSDT" not in url for url in plan.urls)


def test_import_legacy_cache_copies_only_missing_nonempty_payloads(
    tmp_path: Path,
) -> None:
    module = _load_module()
    legacy = tmp_path / "legacy"
    destination = tmp_path / "destination"
    source_payload = legacy / "ab" / ("ab" + "1" * 62 + ".bin")
    empty_payload = legacy / "cd" / ("cd" + "2" * 62 + ".bin")
    existing_payload = destination / "ef" / ("ef" + "3" * 62 + ".bin")
    replacement_source = legacy / "ef" / ("ef" + "3" * 62 + ".bin")
    source_payload.parent.mkdir(parents=True)
    empty_payload.parent.mkdir(parents=True)
    existing_payload.parent.mkdir(parents=True)
    replacement_source.parent.mkdir(parents=True, exist_ok=True)
    source_payload.write_bytes(b"legacy")
    empty_payload.write_bytes(b"")
    existing_payload.write_bytes(b"new")
    replacement_source.write_bytes(b"old")

    copied = module.import_legacy_cache(
        source_root=legacy,
        destination_root=destination,
    )

    assert copied == 1
    assert (destination / source_payload.relative_to(legacy)).read_bytes() == b"legacy"
    assert existing_payload.read_bytes() == b"new"
    assert not (destination / empty_payload.relative_to(legacy)).exists()
