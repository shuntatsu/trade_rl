from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.data.artifacts import write_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.workflows.training_run import execute_training_run


def _dataset() -> MarketDataset:
    n_bars = 40
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )


def test_execute_training_run_trains_serializes_and_publishes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_artifact(dataset_root, _dataset())
    config_path = tmp_path / "train.json"
    _config(config_path)
    store_root = tmp_path / "artifacts"

    result = execute_training_run(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=store_root,
        run_id="tiny-run",
    )

    published = store_root / "runs" / "tiny-run"
    assert result.status == "published"
    assert result.path == published
    assert (published / "members" / "member-000" / "policy.zip").is_file()
    assert (published / "ensemble.json").is_file()
    assert (published / "training-config.json").is_file()
    assert (published / "dataset-reference.json").is_file()
    assert (published / "run.json").is_file()
    pointer = json.loads((store_root / "latest.json").read_text(encoding="utf-8"))
    assert pointer == {"path": "runs/tiny-run", "run_id": "tiny-run"}
