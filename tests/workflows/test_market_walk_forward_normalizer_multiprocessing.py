from __future__ import annotations

import multiprocessing
import threading
import warnings

import numpy as np
import pytest

from tests.support.training_config import complete_execution_config
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.workflows import _market_walk_forward_core as core
from trade_rl.workflows.training_run import TrainingRunConfig


def _dataset() -> MarketDataset:
    n_bars = 64
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = (100.0 + 0.25 * np.arange(n_bars, dtype=np.float64))[:, None]
    return MarketDataset(
        dataset_id="c" * 64,
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


def _run() -> TrainingRunConfig:
    return TrainingRunConfig.from_mapping(
        {
            "schema_version": "training_run_config_v4",
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
                "baseline_underperformance_weight": 0.0,
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
    )


def test_normalizer_parallel_start_method_prefers_forkserver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core.multiprocessing,
        "get_all_start_methods",
        lambda: ["fork", "spawn", "forkserver"],
    )

    resolver = getattr(core, "_normalizer_start_method", lambda: "fork")

    assert resolver() == "forkserver"


def test_normalizer_parallel_start_method_falls_back_to_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core.multiprocessing,
        "get_all_start_methods",
        lambda: ["fork", "spawn"],
    )

    resolver = getattr(core, "_normalizer_start_method", lambda: "fork")

    assert resolver() == "spawn"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the Python 3.12 fork deprecation is POSIX-specific",
)
def test_parallel_normalizer_does_not_fork_a_multithreaded_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADE_RL_PREPROCESS_WORKERS", "2")
    started = threading.Event()
    stop = threading.Event()

    def keep_thread_alive() -> None:
        started.set()
        stop.wait()

    thread = threading.Thread(target=keep_thread_alive, daemon=True)
    thread.start()
    assert started.wait(timeout=5.0)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            core._fit_normalizer(_dataset(), IndexRange(0, 30), _run())
    finally:
        stop.set()
        thread.join(timeout=5.0)

    fork_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "fork" in str(warning.message).lower()
    ]
    assert fork_warnings == []


def test_parallel_normalizer_matches_serial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    run = _run()
    train_range = IndexRange(0, 30)

    monkeypatch.setenv("TRADE_RL_PREPROCESS_WORKERS", "1")
    serial = core._fit_normalizer(dataset, train_range, run)

    monkeypatch.setenv("TRADE_RL_PREPROCESS_WORKERS", "2")
    parallel = core._fit_normalizer(dataset, train_range, run)

    np.testing.assert_array_equal(parallel.mean, serial.mean)
    np.testing.assert_array_equal(parallel.scale, serial.scale)
    assert parallel.digest == serial.digest
