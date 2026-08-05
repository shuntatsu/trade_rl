from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_support.training_config import complete_execution_config
from trade_rl.evaluation.walk_forward.sealed_test import SealedTestLedger
from trade_rl.workflows import market_walk_forward as workflow_module
from trade_rl.workflows._market_walk_forward_core import _experiment_plan_digest
from trade_rl.workflows.market_walk_forward_config import (
    MarketWalkForwardConfig,
    SealedTestLedgerMode,
)


def _candidate_run() -> dict[str, object]:
    return {
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
            "seeds": [0, 1],
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


def _config(tmp_path: Path, mode: str | None) -> MarketWalkForwardConfig:
    payload: dict[str, object] = {
        "workflow": {
            "train_bars": 30,
            "checkpoint_bars": 6,
            "selection_bars": 6,
            "test_bars": 6,
            "purge_bars": 1,
            "max_folds": 1,
        },
        "candidates": [{"name": "ppo", "run": _candidate_run()}],
    }
    if mode is not None:
        payload["sealed_test_ledger_mode"] = mode
    path = tmp_path / f"walk-forward-{mode or 'legacy'}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return MarketWalkForwardConfig.from_json(path, n_bars=64)


def test_legacy_config_normalizes_to_explicit_local_mode(tmp_path: Path) -> None:
    config = _config(tmp_path, None)
    assert config.sealed_test_ledger_mode is SealedTestLedgerMode.LOCAL_EXPLORATORY
    assert config.digest_payload()["sealed_test_ledger_mode"] == "local_exploratory"


def test_ledger_mode_changes_config_and_experiment_identity(tmp_path: Path) -> None:
    local = _config(tmp_path, "local_exploratory")
    durable = _config(tmp_path, "durable_postgres")

    assert local.digest_payload() != durable.digest_payload()
    assert _experiment_plan_digest(local, dataset_id="a" * 64) != (
        _experiment_plan_digest(durable, dataset_id="a" * 64)
    )


def test_local_mode_ignores_database_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADE_RL_DATABASE_URL", "postgresql://must-not-be-used")
    monkeypatch.setattr(
        workflow_module,
        "PostgresSealedTestReservationStore",
        lambda _: pytest.fail("local mode constructed PostgreSQL sealed-test store"),
    )

    ledger = workflow_module._sealed_test_ledger(SealedTestLedgerMode.LOCAL_EXPLORATORY)

    assert isinstance(ledger, SealedTestLedger)


def test_durable_mode_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADE_RL_DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="durable PostgreSQL sealed-test ledger"):
        workflow_module._sealed_test_ledger(SealedTestLedgerMode.DURABLE_POSTGRES)


def test_durable_mode_constructs_dedicated_store_without_runtime_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Store:
        def __init__(self, database_url: str) -> None:
            observed["database_url"] = database_url
            observed["store"] = self

        def migrate(self) -> None:
            pytest.fail("walk-forward runtime must not apply catalog migrations")

    class Ledger:
        def __init__(self, store: object) -> None:
            observed["ledger_store"] = store

    monkeypatch.setenv("TRADE_RL_DATABASE_URL", "postgresql://explicit")
    monkeypatch.setattr(workflow_module, "PostgresSealedTestReservationStore", Store)
    monkeypatch.setattr(workflow_module, "PostgresSealedTestLedger", Ledger)

    ledger = workflow_module._sealed_test_ledger(SealedTestLedgerMode.DURABLE_POSTGRES)

    assert isinstance(ledger, Ledger)
    assert observed["database_url"] == "postgresql://explicit"
    assert observed["ledger_store"] is observed["store"]


def test_unknown_ledger_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed_test_ledger_mode"):
        _config(tmp_path, "ambient")
