from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingCursor,
    build_symbol_triplet_training_plan,
    initial_symbol_triplet_training_cursor,
    write_symbol_triplet_training_cursor,
)

_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "TRXUSDT",
    "ETCUSDT",
)
_SLOT_SYMBOLS = ("SLOT0", "SLOT1", "SLOT2")


def _plan():
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=31)
    return build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )


def _dataset(identity: str) -> MarketDataset:
    n_bars = 8
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(15, "m")
    close = np.stack(
        [100.0 + np.arange(n_bars, dtype=np.float64) + offset for offset in range(3)],
        axis=1,
    )
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=_SLOT_SYMBOLS,
        timestamps=timestamps,
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 3), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 3)),
        tradable=np.ones((n_bars, 3), dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    ).with_content_identity({"identity": identity})


def _run_kwargs(tmp_path: Path, plan: Any) -> dict[str, Any]:
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    base_config_path = tmp_path / "base-config.json"
    base_config_path.write_text("{}\n", encoding="utf-8")
    return {
        "connection": object(),
        "plan": plan,
        "cursor_path": cursor_path,
        "base_config_path": base_config_path,
        "work_root": tmp_path / "triplet-training",
        "symbol_vocabulary": _SYMBOLS,
        "start_time": datetime(2024, 12, 1, tzinfo=UTC),
        "end_time": datetime(2026, 7, 1, tzinfo=UTC),
        "metadata": {symbol: {} for symbol in _SYMBOLS},
        "metadata_evidence_digest": "a" * 64,
    }


def test_runner_builds_current_postgres_stage_and_calls_training_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_runner as module

    plan = _plan()
    kwargs = _run_kwargs(tmp_path, plan)
    calls: list[dict[str, Any]] = []
    sentinel = object()

    monkeypatch.setattr(module, "_training_seeds", lambda _: (0, 1))

    def build_dataset(connection: object, **builder_kwargs: Any) -> MarketDataset:
        calls.append({"connection": connection, **builder_kwargs})
        return _dataset("stage-0")

    def execute_training(**executor_kwargs: Any) -> object:
        calls.append({"executor": executor_kwargs})
        return sentinel

    monkeypatch.setattr(module, "build_postgres_market_dataset", build_dataset)
    monkeypatch.setattr(module, "execute_symbol_triplet_stage_training", execute_training)

    result = module.execute_binance_symbol_triplet_postgres_stage(**kwargs)

    assert result is sentinel
    request = module.current_binance_symbol_triplet_stage_request(
        plan,
        kwargs["cursor_path"],
        kwargs["base_config_path"],
        kwargs["work_root"],
    )
    assert request is not None
    builder_call = calls[0]
    assert builder_call["symbols"] == request.symbols
    assert builder_call["slot_symbols"] == request.slot_symbols
    assert builder_call["symbol_vocabulary"] == _SYMBOLS
    assert builder_call["symbol_triplet_provenance"] == {
        "cycle_index": request.cycle_index,
        "plan_digest": request.plan_digest,
        "request_digest": request.digest,
        "schema_version": "binance_symbol_triplet_stage_provenance_v1",
        "selected_symbols": request.symbols,
        "slot_symbols": request.slot_symbols,
        "stage_id": request.stage_id,
        "stage_index": request.stage_index,
        "train_split_slot": request.train_split_slot,
    }
    stage_root = module.binance_symbol_triplet_stage_root(kwargs["work_root"], request)
    assert (stage_root / "dataset" / "manifest.json").is_file()
    assert (stage_root / "dataset-binding.json").is_file()
    executor_call = calls[1]["executor"]
    assert executor_call["dataset_path"] == stage_root / "dataset"
    assert executor_call["dataset_binding_path"] == stage_root / "dataset-binding.json"
    assert executor_call["stage_config_path"] == stage_root / "training-config.json"
    assert executor_call["completion_path"] == stage_root / "completion.json"
    assert executor_call["previous_completion_path"] is None
    assert executor_call["run_id"].startswith("binance-triplet-stage-0000-")


def test_runner_resumes_existing_bound_dataset_without_rebuilding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_runner as module

    plan = _plan()
    kwargs = _run_kwargs(tmp_path, plan)
    monkeypatch.setattr(module, "_training_seeds", lambda _: (0, 1))
    monkeypatch.setattr(module, "execute_symbol_triplet_stage_training", lambda **_: "ok")
    monkeypatch.setattr(
        module,
        "build_postgres_market_dataset",
        lambda *_args, **_kwargs: _dataset("stable"),
    )

    assert module.execute_binance_symbol_triplet_postgres_stage(**kwargs) == "ok"
    monkeypatch.setattr(
        module,
        "build_postgres_market_dataset",
        lambda *_args, **_kwargs: pytest.fail("bound dataset must be reused"),
    )

    assert module.execute_binance_symbol_triplet_postgres_stage(**kwargs) == "ok"


def test_runner_rejects_orphan_dataset_without_advancing_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_runner as module
    from trade_rl.data import publish_market_dataset_artifact

    plan = _plan()
    kwargs = _run_kwargs(tmp_path, plan)
    monkeypatch.setattr(module, "_training_seeds", lambda _: (0, 1))
    request = module.current_binance_symbol_triplet_stage_request(
        plan,
        kwargs["cursor_path"],
        kwargs["base_config_path"],
        kwargs["work_root"],
    )
    assert request is not None
    stage_root = module.binance_symbol_triplet_stage_root(kwargs["work_root"], request)
    publish_market_dataset_artifact(stage_root / "dataset", _dataset("orphan"))
    before = Path(kwargs["cursor_path"]).read_bytes()

    with pytest.raises(FileExistsError, match="binding"):
        module.execute_binance_symbol_triplet_postgres_stage(**kwargs)

    assert Path(kwargs["cursor_path"]).read_bytes() == before
    assert not (stage_root / "completion.json").exists()


def test_runner_returns_none_for_completed_plan_without_touching_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_runner as module

    plan = _plan()
    kwargs = _run_kwargs(tmp_path, plan)
    complete = SymbolTripletTrainingCursor(
        plan_digest=plan.digest,
        stage_count=plan.stage_count,
        next_stage_index=plan.stage_count,
        last_completed_stage_id=plan.stages[-1].stage_id,
    )
    write_symbol_triplet_training_cursor(kwargs["cursor_path"], complete)
    monkeypatch.setattr(
        module,
        "_training_seeds",
        lambda _: pytest.fail("completed plan must not load training config"),
    )
    monkeypatch.setattr(
        module,
        "build_postgres_market_dataset",
        lambda *_args, **_kwargs: pytest.fail("completed plan must not build data"),
    )

    assert module.execute_binance_symbol_triplet_postgres_stage(**kwargs) is None
