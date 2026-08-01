from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_support.training_config import complete_execution_config
from trade_rl.artifacts.provenance import capture_runtime_provenance
from trade_rl.artifacts.run_manifest import (
    WALK_FORWARD_RUN_MANIFEST_SCHEMA,
    validate_training_run_directory,
    validate_walk_forward_run_directory,
)
from trade_rl.data import write_market_dataset_files
from trade_rl.data.market import MarketDataset
from trade_rl.workflows import market_walk_forward as workflow_module


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
        tick_size=np.full((n_bars, 1), 0.1),
        lot_size=np.full((n_bars, 1), 0.01),
        minimum_notional=np.full((n_bars, 1), 5.0),
    ).with_content_identity()


def _candidate_run() -> dict[str, object]:
    return {
        "schema_version": "training_run_config_v4",
        "git_commit": "b" * 40,
        "git_dirty": False,
        "training": {
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
            "timesteps": 8,
            "gamma": 0.99,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
            "n_epochs": 1,
            "observation_encoder": "flat_mlp",
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
        "exports": {"onnx": False, "torchscript": False},
    }


def _provenance(tmp_path: Path):
    root = tmp_path / "source"
    (root / "trade_rl").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='trade-rl'\n", encoding="utf-8"
    )
    (root / "uv.lock").write_text("test-lock", encoding="utf-8")
    (root / "uv.toml").write_text('required-version = "==0.10.0"\n', encoding="utf-8")
    (root / "trade_rl" / "module.py").write_text("test", encoding="utf-8")
    (root / "examples" / "runner.py").write_text("test", encoding="utf-8")
    return capture_runtime_provenance(
        root,
        git_commit="b" * 40,
        git_dirty=False,
        deterministic_seed_config={"candidate_seeds": (0,)},
        package_versions={"numpy": "test"},
        python_version="3.12.0",
        platform_name="test-platform",
        hardware_name="test-hardware",
    )


def test_market_walk_forward_publishes_dedicated_manifest_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_files(dataset_root, _dataset())
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
    expected_provenance = _provenance(tmp_path)
    provenance_call: dict[str, object] = {}

    def capture_provenance(*args: object, **kwargs: object):
        provenance_call.update(kwargs)
        return expected_provenance

    monkeypatch.setattr(
        workflow_module,
        "capture_runtime_provenance",
        capture_provenance,
        raising=False,
    )

    result = workflow_module.execute_market_walk_forward(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "artifacts",
        run_id="wf-provenance",
    )

    published = result.path
    manifest = validate_walk_forward_run_directory(published)
    provenance_payload = json.loads(
        (published / "provenance.json").read_text(encoding="utf-8")
    )

    assert manifest.schema_version == WALK_FORWARD_RUN_MANIFEST_SCHEMA
    assert manifest.evaluation_digest == result.evaluation_digest
    assert manifest.fold_count == 1
    assert manifest.provenance_digest == expected_provenance.digest
    assert provenance_payload["digest"] == manifest.provenance_digest
    assert provenance_call["git_commit"] == "b" * 40
    assert provenance_call["git_dirty"] is False
    assert manifest.workflow_config_digest != manifest.provenance_digest
    assert manifest.policy_set_digest != manifest.workflow_config_digest
    with pytest.raises(ValueError, match="unsupported training run schema"):
        validate_training_run_directory(published)
