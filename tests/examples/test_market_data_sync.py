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


def test_build_maintained_plan_uses_pipeline_identity() -> None:
    module = _load_module()

    plan = module.build_maintained_plan()

    assert plan.symbols == tuple(module.pipeline._SYMBOLS)
    assert plan.intervals == tuple(module.pipeline._NATIVE_TIMEFRAMES)
    assert plan.start_time == datetime.fromisoformat(
        module.pipeline._START.replace("Z", "+00:00")
    ).astimezone(UTC)
    assert plan.end_time == datetime.fromisoformat(
        module.pipeline._END.replace("Z", "+00:00")
    ).astimezone(UTC)
    assert plan.urls
