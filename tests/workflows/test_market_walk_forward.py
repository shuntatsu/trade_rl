from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.data.artifacts import write_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.workflows.market_walk_forward import execute_market_walk_forward


def _dataset() -> MarketDataset:
    n_bars = 64
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + 0.25 * np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.sin(np.arange(n_bars, dtype=np.float32))[:, None, None],
        global_features=np.cos(np.arange(n_bars, dtype=np.float32))[:, None],
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _candidate_run() -> dict[str, object]:
    return {
        "training": {
            "timesteps": 8,
            "gamma": 0.99,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "asset_set_encoder": False,
            "device": "cpu",
        },
        "environment": {
            "episode_hours": 4.0,
            "decision_hours": 1.0,
            "episode_bars": 4,
            "decision_every": 1,
            "initial_capital": 1_000.0,
            "initial_state_modes": ["cash"],
        },
        "risk": {
            "max_gross": 1.0,
            "max_abs_weight": 1.0,
            "max_turnover": 2.0,
        },
        "reward": {"scale": 1.0},
        "trend": {
            "fast_hours": 1.0,
            "base_hours": 2.0,
            "slow_hours": 3.0,
            "fast_lookback": 1,
            "base_lookback": 2,
            "slow_lookback": 3,
            "mode": "time_series",
        },
        "action": {"alpha_enabled": False, "n_factors": 0},
        "exports": {"onnx": False, "torchscript": False},
    }


def test_market_walk_forward_trains_selects_and_evaluates_sealed_test_once(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_artifact(dataset_root, _dataset())
    config_path = tmp_path / "walk-forward.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow": {
                    "train_bars": 30,
                    "checkpoint_bars": 6,
                    "selection_bars": 6,
                    "test_bars": 6,
                    "purge_bars": 1,
                    "max_folds": 1,
                },
                "minimum_selection_uplift": 0.0,
                "candidates": [{"name": "ppo", "run": _candidate_run()}],
            }
        ),
        encoding="utf-8",
    )

    result = execute_market_walk_forward(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "artifacts",
        run_id="wf-001",
    )

    assert result.status == "published"
    published = tmp_path / "artifacts" / "runs" / "wf-001"
    payload = json.loads((published / "walk-forward.json").read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "a" * 64
    assert len(payload["folds"]) == 1
    assert payload["folds"][0]["test_range"] == [45, 51]
    assert payload["folds"][0]["sealed_test_evaluations"] in (1, 2)
    normalizer = json.loads(
        (published / "fold-000" / "normalizer.json").read_text(encoding="utf-8")
    )
    assert normalizer["absolute_train_range"] == [0, 30]
    assert normalizer["dataset_id"] == "a" * 64
    assert (published / "run.json").is_file()
