from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.strategy_comparison import compare_strategies_by_symbol
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def market() -> MarketDataset:
    n = 8
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
            [115.0, 85.0],
            [120.0, 80.0],
            [125.0, 75.0],
            [130.0, 70.0],
        ]
    )
    signal = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def run_config() -> dict[str, object]:
    return {
        "signal_name": "signal",
        "feature_names": ["signal"],
        "fit_symbol_names": ["BTCUSDT"],
        "fit_cutoff": "2026-01-01T04:00:00",
        "evaluation_start": "2026-01-01T04:00:00",
        "evaluation_stop_exclusive": "2026-01-01T07:00:00",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
        "gross_budget": 0.5,
        "initial_capital": 1000.0,
    }


def test_run_candidate_artifact_writes_summary_and_raw_returns(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation import candidate_run

    dataset = market()
    comparison = compare_strategies_by_symbol(
        dataset,
        {
            "cash": ConstantIntentStrategy(PositionIntent.FLAT),
            "long": ConstantIntentStrategy(PositionIntent.LONG),
        },
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: dataset,
    )

    def fake_suite(loaded, config, **kwargs):
        calls["dataset"] = loaded
        calls["config"] = config
        calls["kwargs"] = kwargs
        return comparison

    monkeypatch.setattr(candidate_run, "run_lean_candidate_suite", fake_suite)

    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(run_config()), encoding="utf-8")
    output = tmp_path / "result"

    artifact = candidate_run.run_candidate_artifact(
        dataset_root=tmp_path / "dataset",
        config_path=config_path,
        output_root=output,
    )

    assert artifact.root == output
    assert artifact.summary_path == output / "summary.json"
    assert artifact.returns_path == output / "returns.npz"
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "lean_candidate_result_v1"
    assert summary["dataset_id"] == dataset.dataset_id
    assert summary["dataset_artifact"] == {
        "schema_version": "market_dataset_artifact_v3",
        "artifact_digest": "d" * 64,
    }
    assert summary["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert summary["candidate_config"] == {
        "signal_name": "signal",
        "signal_index": 0,
        "feature_names": ["signal"],
        "feature_indices": [0],
        "fit_symbol_names": ["BTCUSDT"],
        "fit_symbol_indices": [0],
        "fit_cutoff": "2026-01-01T04:00:00.000000000",
        "rule_entry_threshold": 0.10,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 7,
    }
    assert summary["evaluation"] == {
        "start": "2026-01-01T04:00:00.000000000",
        "stop_exclusive": "2026-01-01T07:00:00.000000000",
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "execution_overlay": "zero_overlay_dataset_fields_authoritative",
    }
    assert [item["symbol"] for item in summary["by_symbol"]] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert summary["by_symbol"][0]["strategies"][0]["name"] == "cash"
    assert summary["by_symbol"][0]["strategies"][1]["name"] == "long"

    with np.load(artifact.returns_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "symbol_0_strategy_0",
            "symbol_0_strategy_1",
            "symbol_1_strategy_0",
            "symbol_1_strategy_1",
        }
        assert arrays["symbol_0_strategy_1"].shape == (3,)

    lean_config = calls["config"]
    assert getattr(lean_config, "signal_index") == 0
    assert getattr(lean_config, "feature_indices") == (0,)
    assert getattr(lean_config, "fit_symbol_indices") == (0,)
    assert calls["kwargs"] == {
        "start_index": 4,
        "stop_index": 7,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "execution_cost": None,
        "risk": None,
    }


def test_run_candidate_artifact_rejects_unknown_config_keys(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation import candidate_run

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    raw = run_config()
    raw["hidden_override"] = 1
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config keys"):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=tmp_path / "result",
        )


def test_run_candidate_artifact_rejects_unknown_fit_symbol(
    tmp_path,
    monkeypatch,
) -> None:
    from trade_rl.evaluation import candidate_run

    monkeypatch.setattr(
        candidate_run,
        "inspect_published_market_dataset_artifact",
        lambda path: SimpleNamespace(
            schema_version="market_dataset_artifact_v3",
            artifact_digest="d" * 64,
        ),
    )
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    raw = run_config()
    raw["fit_symbol_names"] = ["UNKNOWN"]
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fit symbol name: UNKNOWN"):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=tmp_path / "result",
        )


def test_run_candidate_artifact_refuses_overwrite(tmp_path, monkeypatch) -> None:
    from trade_rl.evaluation import candidate_run

    output = tmp_path / "result"
    output.mkdir()
    monkeypatch.setattr(
        candidate_run,
        "load_market_dataset_artifact",
        lambda path: market(),
    )
    config_path = tmp_path / "run.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        candidate_run.run_candidate_artifact(
            dataset_root=tmp_path / "dataset",
            config_path=config_path,
            output_root=output,
        )
