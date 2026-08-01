from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_support.training_config import complete_execution_config
from trade_rl.data import write_market_dataset_files
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution_promotion import load_execution_evidence
from trade_rl.workflows.training_run import TrainingRunConfig, execute_training_run


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
    ).with_content_identity()


def _config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "training_run_config_v4",
                "training": {
                    "timesteps": 8,
                    "gamma": 0.99,
                    "seeds": [0],
                    "n_steps": 8,
                    "batch_size": 8,
                    "n_epochs": 1,
                    "observation_encoder": "flat_mlp",
                    "policy_actor_head": "standard_continuous_v1",
                    "hierarchical_gate_temperature": 1.0,
                    "behavior_cloning_gate_loss_weight": 1.0,
                    "behavior_cloning_target_loss_weight": 1.0,
                    "behavior_cloning_composed_loss_weight": 1.0,
                    "behavior_cloning_gate_change_threshold": 0.05,
                    "behavior_cloning_max_positive_class_weight": 20.0,
                    "behavior_cloning_min_gate_precision": 0.0,
                    "behavior_cloning_min_gate_recall": 0.0,
                    "behavior_cloning_max_active_target_rmse": 1.0,
                    "behavior_cloning_min_activity_ratio": 0.0,
                    "behavior_cloning_max_activity_ratio": 1.0,
                    "behavior_cloning_min_causal_holdout_trades": 0,
                    "behavior_cloning_max_causal_holdout_regret": 0.0,
                    "behavior_cloning_causal_holdout_bootstrap_resamples": 2_000,
                    "behavior_cloning_causal_holdout_confidence_level": 0.95,
                    "device": "cpu",
                },
                "environment": {
                    "episode_hours": 4.0,
                    "decision_hours": 1.0,
                    "episode_bars": 4,
                    "decision_every": 1,
                    "initial_capital": 1_000.0,
                    "initial_state_modes": ["cash"],
                    "require_full_reward_preroll": True,
                },
                "execution": complete_execution_config(),
                "risk": {
                    "max_gross": 1.0,
                    "max_abs_weight": 1.0,
                    "max_turnover": 2.0,
                },
                "reward": {
                    "scale": 1.0,
                    "baseline_window_hours": 4.0,
                    "baseline_minimum_history_hours": 4.0,
                },
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
                "exports": {
                    "onnx": False,
                    "structured_torchscript": False,
                    "torchscript": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_execute_training_run_trains_serializes_and_publishes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_files(dataset_root, _dataset())
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
    assert (published / "policy-loader.json").is_file()
    execution_evidence = load_execution_evidence(published / "execution-evidence.json")
    assert execution_evidence.dataset_id == result.dataset_id
    assert execution_evidence.complete_order_evidence is False
    assert execution_evidence.order_event_count == 0
    environment = json.loads(
        (published / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["terminal_accounting_mode"] == "mark_to_market"
    assert environment["environment"]["liquidate_on_end"] is False
    loader = json.loads((published / "policy-loader.json").read_text(encoding="utf-8"))
    assert loader == {
        "algorithm": "ppo",
        "members": ["members/member-000/policy.zip"],
        "schema_version": "sb3_policy_loader_v1",
    }
    assert (published / "run.json").is_file()
    pointer = json.loads((store_root / "latest.json").read_text(encoding="utf-8"))
    assert pointer == {"path": "runs/tiny-run", "run_id": "tiny-run"}


def test_execute_training_run_uses_declared_provenance_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_files(dataset_root, _dataset())
    config_path = tmp_path / "train.json"
    _config(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["git_commit"] = "c" * 40
    config["git_dirty"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr("trade_rl.artifacts.provenance._git", lambda *_args: None)

    result = execute_training_run(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "artifacts",
        run_id="packaged-source-run",
    )

    provenance = json.loads(
        (result.path / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["git_commit"] == "c" * 40
    assert provenance["git_dirty"] is False


def test_v2_training_config_is_rejected_with_migration_message(tmp_path: Path) -> None:
    config_path = tmp_path / "v2.json"
    _config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "training_run_config_v2"

    with pytest.raises(ValueError, match="training_run_config_v4"):
        TrainingRunConfig.from_mapping(payload)


def test_v4_requires_explicit_actor_head(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-actor.json"
    _config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    del payload["training"]["policy_actor_head"]

    with pytest.raises(ValueError, match="policy_actor_head"):
        TrainingRunConfig.from_mapping(payload)
