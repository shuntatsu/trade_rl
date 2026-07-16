from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from trade_rl.data import write_market_dataset_files
from trade_rl.data.market import MarketDataset
from trade_rl.workflows.market_walk_forward import (
    _experiment_plan_digest,
    execute_market_walk_forward,
)
from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig


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
    ).with_content_identity()


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


def test_experiment_plan_binds_workflow_and_complete_candidate_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "walk-forward.json"
    path.write_text(
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
                "checkpoint_finalists_per_seed": 2,
                "candidates": [{"name": "ppo", "run": _candidate_run()}],
            }
        ),
        encoding="utf-8",
    )
    config = MarketWalkForwardConfig.from_json(path, n_bars=64)
    original = _experiment_plan_digest(config, dataset_id="a" * 64)
    changed_workflow = replace(
        config,
        workflow=replace(config.workflow, test_bars=7),
    )
    changed_training = replace(
        config,
        candidates=(
            replace(
                config.candidates[0],
                run=replace(
                    config.candidates[0].run,
                    training=replace(
                        config.candidates[0].run.training,
                        seeds=(0, 1),
                    ),
                ),
            ),
        ),
    )

    assert (
        _experiment_plan_digest(
            changed_workflow,
            dataset_id="a" * 64,
        )
        != original
    )
    assert (
        _experiment_plan_digest(
            changed_training,
            dataset_id="a" * 64,
        )
        != original
    )


def test_market_walk_forward_trains_selects_and_evaluates_sealed_test_once(
    tmp_path: Path,
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

    result = execute_market_walk_forward(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "artifacts",
        run_id="wf-001",
    )

    assert result.status == "published"
    published = tmp_path / "artifacts" / "runs" / "wf-001"
    payload = json.loads((published / "walk-forward.json").read_text(encoding="utf-8"))
    assert payload["dataset_id"] == _dataset().dataset_id
    assert payload["schema_version"] == "market_walk_forward_run_v5_deployable_ensemble"
    assert len(payload["experiment_plan_digest"]) == 64
    assert len(payload["folds"]) == 1
    assert payload["folds"][0]["test_range"] == [45, 51]
    assert payload["folds"][0]["sealed_test_evaluations"] in (1, 2)
    sealed_access = payload["folds"][0]["sealed_test_access"]
    assert sealed_access["experiment_plan_digest"] == payload["experiment_plan_digest"]
    assert len(sealed_access["access_digest"]) == 64
    assert sealed_access["test_range"] == [45, 51]
    assert payload["folds"][0]["schema_version"] == (
        "market_walk_forward_fold_v4_deployable_ensemble"
    )
    assert payload["folds"][0]["selected_member_seeds"] == [0]
    assert len(payload["folds"][0]["selected_member_policy_digests"]) == 1
    assert len(payload["folds"][0]["seed_finalists"]) == 1
    finalist = payload["folds"][0]["seed_finalists"][0]
    assert finalist["seed"] == 0
    assert len(finalist["checkpoint_evaluation_digest"]) == 64
    assert len(finalist["selection_evaluation_digest"]) == 64
    checkpoint_selection = json.loads(
        (
            published / "fold-000" / "candidates" / "ppo" / "checkpoint-selection.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint_selection["schema_version"] == (
        "checkpoint_selection_v2_seed_aware"
    )
    assert len(checkpoint_selection["seed_finalists"]) == 1
    normalizer = json.loads(
        (published / "fold-000" / "normalizer.json").read_text(encoding="utf-8")
    )
    assert normalizer["absolute_train_range"][1] == 30
    assert normalizer["absolute_train_range"][0] > 0
    assert normalizer["dataset_id"] == _dataset().dataset_id
    assert (published / "run.json").is_file()


def _sequence_dataset() -> MarketDataset:
    n_bars = 320
    timestamps = np.datetime64("2026-01-01T00:15:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(15, "m")
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(0.0001 * phase))[:, None]
    feature_names = (
        "15m__feature",
        "1h__feature",
        "4h__feature",
        "1d__feature",
    )
    features = np.stack(
        tuple(np.sin(phase / divisor) for divisor in (5.0, 7.0, 11.0, 17.0)),
        axis=1,
    )[:, None, :].astype(np.float32)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=np.cos(phase / 13.0)[:, None].astype(np.float32),
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, len(feature_names)), dtype=np.bool_),
        feature_names=feature_names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    ).with_content_identity()


def _sequence_candidate_config():
    from trade_rl.workflows.training_run import TrainingRunConfig

    return TrainingRunConfig.from_mapping(
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
                "policy": "MultiInputPolicy",
                "sequence_encoder": True,
            },
            "environment": {
                "episode_hours": 4.0,
                "decision_hours": 0.25,
                "episode_bars": 16,
                "decision_every": 1,
                "initial_capital": 1_000.0,
                "initial_state_modes": ["cash"],
                "structured_sequence_observation": True,
                "sequence_windows": [
                    ["15m", 4],
                    ["1h", 3],
                    ["4h", 2],
                    ["1d", 2],
                ],
            },
            "risk": {
                "max_gross": 1.0,
                "max_abs_weight": 1.0,
                "max_turnover": None,
            },
            "reward": {
                "scale": 1.0,
                "baseline_window_hours": 4.0,
                "baseline_minimum_history_hours": 4.0,
                "baseline_underperformance_weight": 0.1,
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
            "action": {
                "mode": "target_weight",
                "alpha_enabled": False,
                "risk_tilt_enabled": False,
                "n_factors": 0,
                "target_weight_count": 1,
            },
            "exports": {"onnx": False, "torchscript": False},
        }
    )


def test_structured_training_view_preserves_exact_sequence_and_reward_preroll() -> None:
    from trade_rl.evaluation.walk_forward.folds import IndexRange
    from trade_rl.workflows.market_walk_forward import (
        _training_view,
        _training_view_bounds,
    )
    from trade_rl.workflows.walk_forward_evaluation import minimum_environment_start

    dataset = _sequence_dataset()
    run = _sequence_candidate_config()
    train_range = IndexRange(150, 220)

    view_start, view_stop = _training_view_bounds(dataset, train_range, run)
    training_dataset = _training_view(dataset, train_range, run)
    minimum = minimum_environment_start(training_dataset, run)

    assert view_stop == train_range.stop
    assert minimum == train_range.start - view_start


def test_structured_walk_forward_fits_flat_snapshot_normalizer_train_only() -> None:
    from trade_rl.evaluation.walk_forward.folds import IndexRange
    from trade_rl.rl.observations import observation_layout
    from trade_rl.workflows.market_walk_forward import _fit_normalizer, _training_view

    dataset = _sequence_dataset()
    run = _sequence_candidate_config()
    train_range = IndexRange(150, 220)

    normalizer = _fit_normalizer(dataset, train_range, run)
    training_dataset = _training_view(dataset, train_range, run)
    expected = observation_layout(
        training_dataset,
        action_size=run.action.size,
        n_factors=run.action.n_factors,
        finite_horizon=run.environment.finite_horizon_observation,
    )

    assert normalizer.size == expected.size
    assert normalizer.absolute_train_start == train_range.start
    assert normalizer.absolute_train_end == train_range.stop


def test_structured_walk_forward_fits_sequence_normalizer_on_exact_train_range() -> (
    None
):
    from trade_rl.evaluation.walk_forward.folds import IndexRange
    from trade_rl.workflows.market_walk_forward import (
        _fit_sequence_normalizer,
        _training_view_bounds,
    )

    dataset = _sequence_dataset()
    run = _sequence_candidate_config()
    train_range = IndexRange(150, 220)
    view_start, _ = _training_view_bounds(dataset, train_range, run)

    normalizer = _fit_sequence_normalizer(dataset, train_range, run)

    assert normalizer is not None
    assert normalizer.train_start == train_range.start - view_start
    assert normalizer.train_end == train_range.stop - view_start
    assert normalizer.source_dataset_id == dataset.dataset_id


def test_structured_walk_forward_trains_three_seed_ensemble_end_to_end(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "sequence-dataset"
    dataset = _sequence_dataset()
    write_market_dataset_files(dataset_root, dataset)
    run = _sequence_candidate_config()
    run = replace(
        run,
        training=replace(
            run.training,
            seeds=(0, 1, 2),
            checkpoint_interval_steps=4,
            max_checkpoints=2,
        ),
    )
    config_path = tmp_path / "sequence-walk-forward.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow": {
                    "train_bars": 220,
                    "checkpoint_bars": 20,
                    "selection_bars": 20,
                    "test_bars": 20,
                    "purge_bars": 4,
                    "max_folds": 1,
                },
                "minimum_selection_uplift": 0.0,
                "candidates": [{"name": "sequence-ppo", "run": run.digest_payload()}],
            }
        ),
        encoding="utf-8",
    )

    result = execute_market_walk_forward(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "sequence-artifacts",
        run_id="wf-sequence-001",
    )

    assert result.status == "published"
    published = result.path
    payload = json.loads((published / "walk-forward.json").read_text(encoding="utf-8"))
    fold = payload["folds"][0]
    finalists = fold["seed_finalists"]
    assert [item["seed"] for item in finalists] == [0, 1, 2]
    assert all(len(item["policy_digest"]) == 64 for item in finalists)
    aggregate = fold["candidate_aggregates"][0]
    assert aggregate["configuration"] == "sequence-ppo"
    assert aggregate["seed_count"] == 3
    if fold["selected_configuration"] == "sequence-ppo":
        assert fold["selected_member_seeds"] == [0, 1, 2]
        assert len(fold["selected_member_policy_digests"]) == 3
        assert fold["sealed_test_evaluations"] == 2
    else:
        assert fold["selected_configuration"] == "baseline"
        assert fold["selected_member_seeds"] == []
        assert fold["selected_member_policy_digests"] == []
        assert fold["sealed_test_evaluations"] == 1
    assert (published / "fold-000" / "sequence-normalizer-sequence-ppo.json").is_file()
    checkpoint_selection = json.loads(
        (
            published
            / "fold-000"
            / "candidates"
            / "sequence-ppo"
            / "checkpoint-selection.json"
        ).read_text(encoding="utf-8")
    )
    assert [item["seed"] for item in checkpoint_selection["seed_finalists"]] == [
        0,
        1,
        2,
    ]
