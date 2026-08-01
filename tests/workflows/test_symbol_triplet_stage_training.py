from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from test_support.training_config import complete_execution_config
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.market import MarketDataset
from trade_rl.rl.checkpointing import publish_checkpoint
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    build_symbol_triplet_stage_request,
)
from trade_rl.workflows.symbol_triplet_stage_training import (
    build_symbol_triplet_stage_dataset_binding,
    execute_symbol_triplet_stage_training,
    load_symbol_triplet_stage_dataset_binding,
    write_symbol_triplet_stage_dataset_binding,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    build_symbol_triplet_training_plan,
    initial_symbol_triplet_training_cursor,
    load_symbol_triplet_training_cursor,
    write_symbol_triplet_training_cursor,
)
from trade_rl.workflows.training_run import TrainingRunConfig, TrainingRunResult

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
_SEEDS = (0, 1)


class _FakePolicy:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(f"policy-{self.seed}".encode())


def _plan():
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=31)
    return build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )


def _dataset(dataset_id_seed: str) -> MarketDataset:
    n_bars = 40
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.stack(
        [100.0 + np.arange(n_bars, dtype=np.float64) + offset for offset in range(3)],
        axis=1,
    )
    return MarketDataset(
        dataset_id=dataset_id_seed * 64,
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
        periods_per_year=8_760,
    ).with_content_identity({"dataset_id_seed": dataset_id_seed})


def _write_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "training_run_config_v4",
                "training": {
                    "timesteps": 8,
                    "gamma": 0.99,
                    "seeds": list(_SEEDS),
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
                "action": {
                    "alpha_enabled": False,
                    "mode": "target_weight",
                    "n_factors": 0,
                    "risk_tilt_enabled": False,
                    "target_weight_count": 3,
                },
                "exports": {
                    "onnx": False,
                    "structured_torchscript": False,
                    "torchscript": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_dataset_binding(
    *,
    plan: Any,
    cursor: Any,
    dataset_path: Path,
    output_path: Path,
    previous_completion: Any = None,
) -> Path:
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=previous_completion,
    )
    assert request is not None
    binding = build_symbol_triplet_stage_dataset_binding(
        request,
        dataset_path=dataset_path,
        selected_symbols=request.symbols,
    )
    return write_symbol_triplet_stage_dataset_binding(output_path, binding)


def _fake_training_executor(
    calls: list[TrainingRunConfig],
    *,
    omit_seed: int | None = None,
):
    def execute(**kwargs: Any) -> TrainingRunResult:
        config = TrainingRunConfig.from_json(kwargs["config_path"])
        calls.append(config)
        dataset = load_market_dataset_artifact(kwargs["dataset_path"])
        run_path = Path(kwargs["store_root"]) / "runs" / str(kwargs["run_id"])
        run_path.mkdir(parents=True)
        environment_digest = content_digest(
            {"dataset_id": dataset.dataset_id, "schema_version": "fake_env_v1"}
        )
        training_config_digest = content_digest(config.training.digest_payload())
        members: list[dict[str, object]] = []
        for index, seed in enumerate(config.training.seeds):
            members.append({"checkpoint_digest": str(seed) * 64, "seed": seed})
            member_root = run_path / "members" / f"member-{index:03d}"
            member_root.mkdir(parents=True)
            (member_root / "policy.zip").write_bytes(f"final-{seed}".encode())
            if seed == omit_seed:
                continue
            publish_checkpoint(
                model=_FakePolicy(seed),
                checkpoint_root=member_root / "checkpoints",
                algorithm=config.training.algorithm,
                seed=seed,
                requested_timestep=config.training.timesteps,
                observed_timestep=config.training.timesteps,
                environment_digest=environment_digest,
                training_config_digest=training_config_digest,
            )
        (run_path / "ensemble.json").write_bytes(
            canonical_json_bytes(
                {
                    "actual_timesteps": config.training.timesteps,
                    "environment_digest": environment_digest,
                    "expected_members": len(config.training.seeds),
                    "members": members,
                    "training_config_digest": training_config_digest,
                }
            )
        )
        return TrainingRunResult(
            run_id=str(kwargs["run_id"]),
            status="published",
            path=run_path,
            run_digest=content_digest({"run_id": kwargs["run_id"]}),
            policy_digest=content_digest({"members": members}),
            dataset_id=dataset.dataset_id,
            run_kind="research_exploratory",
        )

    return execute


def _validator(dataset_id: str):
    return lambda _: SimpleNamespace(
        dataset_id=dataset_id,
        digest=content_digest({"dataset_id": dataset_id}),
        run_kind="research_exploratory",
    )


def test_stage_dataset_binding_round_trips_and_rejects_wrong_symbols(
    tmp_path: Path,
) -> None:
    plan = _plan()
    cursor = initial_symbol_triplet_training_cursor(plan)
    dataset_path = tmp_path / "dataset"
    write_market_dataset_files(dataset_path, _dataset("a"))
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=None,
    )
    assert request is not None

    binding = build_symbol_triplet_stage_dataset_binding(
        request,
        dataset_path=dataset_path,
        selected_symbols=request.symbols,
    )
    path = write_symbol_triplet_stage_dataset_binding(
        tmp_path / "binding.json", binding
    )

    assert (
        load_symbol_triplet_stage_dataset_binding(
            path,
            request=request,
            dataset_path=dataset_path,
        )
        == binding
    )
    with pytest.raises(ValueError, match="selected symbols"):
        build_symbol_triplet_stage_dataset_binding(
            request,
            dataset_path=dataset_path,
            selected_symbols=tuple(reversed(request.symbols)),
        )


def test_two_training_stages_transfer_only_previous_seed_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_training as module

    plan = _plan()
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    config_path = _write_config(tmp_path / "base-config.json")
    calls: list[TrainingRunConfig] = []
    monkeypatch.setattr(module, "execute_training_run", _fake_training_executor(calls))

    dataset0_path = tmp_path / "dataset-0"
    write_market_dataset_files(dataset0_path, _dataset("a"))
    cursor0 = load_symbol_triplet_training_cursor(cursor_path, plan=plan)
    binding0_path = _write_dataset_binding(
        plan=plan,
        cursor=cursor0,
        dataset_path=dataset0_path,
        output_path=tmp_path / "binding-0.json",
    )
    monkeypatch.setattr(
        module,
        "validate_training_run_directory",
        _validator(load_market_dataset_artifact(dataset0_path).dataset_id),
    )

    stage0 = execute_symbol_triplet_stage_training(
        plan=plan,
        cursor_path=cursor_path,
        previous_completion_path=None,
        dataset_path=dataset0_path,
        dataset_binding_path=binding0_path,
        base_config_path=config_path,
        stage_config_path=tmp_path / "stage-0-config.json",
        store_root=tmp_path / "artifacts-0",
        run_id="triplet-stage-000",
        completion_path=tmp_path / "completion-0.json",
        stage_state_root=tmp_path / "stage-state",
    )

    assert stage0 is not None
    assert stage0.cursor.next_stage_index == 1
    assert calls[0].transfer_checkpoints == ()
    assert TrainingRunConfig.from_json(stage0.stage_config_path) == calls[0]

    dataset1_path = tmp_path / "dataset-1"
    write_market_dataset_files(dataset1_path, _dataset("b"))
    binding1_path = _write_dataset_binding(
        plan=plan,
        cursor=stage0.cursor,
        dataset_path=dataset1_path,
        output_path=tmp_path / "binding-1.json",
        previous_completion=stage0.completion,
    )
    monkeypatch.setattr(
        module,
        "validate_training_run_directory",
        _validator(load_market_dataset_artifact(dataset1_path).dataset_id),
    )

    stage1 = execute_symbol_triplet_stage_training(
        plan=plan,
        cursor_path=cursor_path,
        previous_completion_path=None,
        dataset_path=dataset1_path,
        dataset_binding_path=binding1_path,
        base_config_path=config_path,
        stage_config_path=tmp_path / "stage-1-config.json",
        store_root=tmp_path / "artifacts-1",
        run_id="triplet-stage-001",
        completion_path=tmp_path / "completion-1.json",
        stage_state_root=tmp_path / "stage-state",
    )

    assert stage1 is not None
    assert stage1.cursor.next_stage_index == 2
    assert dict(calls[1].transfer_checkpoints) == {
        checkpoint.seed: checkpoint.checkpoint_root
        for checkpoint in stage0.completion.checkpoints
    }
    assert all(
        checkpoint.checkpoint_digest
        == stage0.completion.checkpoints[index].checkpoint_digest
        for index, checkpoint in enumerate(stage1.request.transfer_checkpoints)
    )


def test_missing_final_seed_checkpoint_does_not_advance_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_training as module

    plan = _plan()
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    initial_bytes = cursor_path.read_bytes()
    config_path = _write_config(tmp_path / "base-config.json")
    dataset_path = tmp_path / "dataset"
    write_market_dataset_files(dataset_path, _dataset("c"))
    cursor = load_symbol_triplet_training_cursor(cursor_path, plan=plan)
    binding_path = _write_dataset_binding(
        plan=plan,
        cursor=cursor,
        dataset_path=dataset_path,
        output_path=tmp_path / "binding.json",
    )
    calls: list[TrainingRunConfig] = []
    monkeypatch.setattr(
        module,
        "execute_training_run",
        _fake_training_executor(calls, omit_seed=1),
    )
    monkeypatch.setattr(
        module,
        "validate_training_run_directory",
        _validator(load_market_dataset_artifact(dataset_path).dataset_id),
    )

    with pytest.raises(RuntimeError, match="final checkpoint"):
        execute_symbol_triplet_stage_training(
            plan=plan,
            cursor_path=cursor_path,
            previous_completion_path=None,
            dataset_path=dataset_path,
            dataset_binding_path=binding_path,
            base_config_path=config_path,
            stage_config_path=tmp_path / "stage-config.json",
            store_root=tmp_path / "artifacts",
            run_id="triplet-stage-000",
            completion_path=tmp_path / "completion.json",
            stage_state_root=tmp_path / "stage-state",
        )

    assert cursor_path.read_bytes() == initial_bytes
    assert not (tmp_path / "completion.json").exists()


def test_stage_training_rejects_dataset_binding_before_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.symbol_triplet_stage_training as module

    plan = _plan()
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    dataset_a_path = tmp_path / "dataset-a"
    dataset_b_path = tmp_path / "dataset-b"
    write_market_dataset_files(dataset_a_path, _dataset("d"))
    write_market_dataset_files(dataset_b_path, _dataset("e"))
    cursor = load_symbol_triplet_training_cursor(cursor_path, plan=plan)
    binding_path = _write_dataset_binding(
        plan=plan,
        cursor=cursor,
        dataset_path=dataset_a_path,
        output_path=tmp_path / "binding.json",
    )

    def forbidden(**_: Any) -> TrainingRunResult:
        raise AssertionError("training executor must not run")

    monkeypatch.setattr(module, "execute_training_run", forbidden)
    with pytest.raises(ValueError, match="dataset identity"):
        execute_symbol_triplet_stage_training(
            plan=plan,
            cursor_path=cursor_path,
            previous_completion_path=None,
            dataset_path=dataset_b_path,
            dataset_binding_path=binding_path,
            base_config_path=_write_config(tmp_path / "config.json"),
            stage_config_path=tmp_path / "stage-config.json",
            store_root=tmp_path / "artifacts",
            run_id="triplet-stage-000",
            completion_path=tmp_path / "completion.json",
        )
