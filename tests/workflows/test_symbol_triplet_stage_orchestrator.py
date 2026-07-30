from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.rl.checkpointing import publish_checkpoint
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    build_symbol_triplet_stage_request,
    commit_symbol_triplet_stage_completion,
    load_symbol_triplet_stage_completion,
    training_config_for_symbol_triplet_stage,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    build_symbol_triplet_training_plan,
    current_symbol_triplet_training_stage,
    initial_symbol_triplet_training_cursor,
    load_symbol_triplet_training_cursor,
    write_symbol_triplet_training_cursor,
)
from trade_rl.workflows.training_run import TrainingRunConfig

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
    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_bytes(b"policy")


def _plan(*, seed: int = 31):
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=seed)
    return build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=_SLOT_SYMBOLS,
    )


def _training_config() -> TrainingRunConfig:
    return TrainingRunConfig.from_mapping(
        {
            "schema_version": "training_run_config_v3",
            "training": {
                "timesteps": 8,
                "gamma": 0.99,
                "seeds": list(_SEEDS),
                "n_steps": 8,
                "batch_size": 8,
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
            },
            "environment": {
                "episode_bars": 4,
                "decision_every": 1,
                "initial_capital": 1_000.0,
            },
            "risk": {},
            "reward": {},
            "trend": {
                "fast_lookback": 1,
                "base_lookback": 2,
                "slow_lookback": 3,
            },
            "action": {"alpha_enabled": False, "n_factors": 0},
        }
    )


def _checkpoint_roots(tmp_path: Path, *, environment_digest: str) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for seed in _SEEDS:
        manifest = publish_checkpoint(
            model=_FakePolicy(),
            checkpoint_root=tmp_path / f"seed-{seed}",
            algorithm="ppo",
            seed=seed,
            requested_timestep=8,
            observed_timestep=8,
            environment_digest=environment_digest,
            training_config_digest="3" * 64,
        )
        roots[seed] = manifest.policy_path.parent
    return roots


def test_initial_stage_request_has_no_transfer_and_stable_slot_binding() -> None:
    plan = _plan()
    cursor = initial_symbol_triplet_training_cursor(plan)

    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=None,
    )

    assert request is not None
    assert request.stage_id == plan.stages[0].stage_id
    assert request.stage_index == 0
    assert request.slot_bindings == tuple(zip(_SLOT_SYMBOLS, request.symbols, strict=True))
    assert request.transfer_checkpoints == ()
    assert training_config_for_symbol_triplet_stage(
        _training_config(), request
    ).transfer_checkpoints == ()


def test_valid_completion_advances_cursor_and_feeds_next_stage(
    tmp_path: Path,
) -> None:
    plan = _plan()
    cursor_path = write_symbol_triplet_training_cursor(
        tmp_path / "cursor.json",
        initial_symbol_triplet_training_cursor(plan),
    )
    cursor = load_symbol_triplet_training_cursor(cursor_path, plan=plan)
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=None,
    )
    assert request is not None
    checkpoint_roots = _checkpoint_roots(
        tmp_path / "stage-000",
        environment_digest="1" * 64,
    )

    completion_path = tmp_path / "completion-000.json"
    completion, advanced = commit_symbol_triplet_stage_completion(
        plan,
        cursor,
        request=request,
        checkpoint_roots=checkpoint_roots,
        completion_path=completion_path,
        cursor_path=cursor_path,
    )

    assert advanced.next_stage_index == 1
    assert load_symbol_triplet_training_cursor(cursor_path, plan=plan) == advanced
    assert load_symbol_triplet_stage_completion(completion_path, plan=plan) == completion

    next_request = build_symbol_triplet_stage_request(
        plan,
        advanced,
        training_seeds=_SEEDS,
        previous_completion=completion,
    )
    assert next_request is not None
    assert next_request.stage_id == plan.stages[1].stage_id
    assert tuple(ref.seed for ref in next_request.transfer_checkpoints) == _SEEDS
    assert dict(
        training_config_for_symbol_triplet_stage(
            _training_config(), next_request
        ).transfer_checkpoints
    ) == checkpoint_roots


def test_completion_rejects_missing_seed_without_advancing_cursor(
    tmp_path: Path,
) -> None:
    plan = _plan()
    cursor = initial_symbol_triplet_training_cursor(plan)
    cursor_path = write_symbol_triplet_training_cursor(tmp_path / "cursor.json", cursor)
    original_bytes = cursor_path.read_bytes()
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=None,
    )
    assert request is not None
    roots = _checkpoint_roots(tmp_path / "stage-000", environment_digest="1" * 64)
    roots.pop(1)

    with pytest.raises(ValueError, match="checkpoint seeds"):
        commit_symbol_triplet_stage_completion(
            plan,
            cursor,
            request=request,
            checkpoint_roots=roots,
            completion_path=tmp_path / "completion.json",
            cursor_path=cursor_path,
        )

    assert cursor_path.read_bytes() == original_bytes
    assert not (tmp_path / "completion.json").exists()


def test_next_stage_rejects_cross_plan_or_stale_completion(tmp_path: Path) -> None:
    plan = _plan(seed=31)
    other_plan = _plan(seed=32)
    cursor = initial_symbol_triplet_training_cursor(plan)
    request = build_symbol_triplet_stage_request(
        plan,
        cursor,
        training_seeds=_SEEDS,
        previous_completion=None,
    )
    assert request is not None
    completion, advanced = commit_symbol_triplet_stage_completion(
        plan,
        cursor,
        request=request,
        checkpoint_roots=_checkpoint_roots(
            tmp_path / "stage-000",
            environment_digest="1" * 64,
        ),
        completion_path=tmp_path / "completion.json",
        cursor_path=tmp_path / "cursor.json",
    )

    with pytest.raises(ValueError, match="previous stage"):
        build_symbol_triplet_stage_request(
            other_plan,
            initial_symbol_triplet_training_cursor(other_plan),
            training_seeds=_SEEDS,
            previous_completion=completion,
        )

    replayed = replace(completion, stage_id=plan.stages[1].stage_id, digest="")
    with pytest.raises(ValueError, match="previous stage"):
        build_symbol_triplet_stage_request(
            plan,
            advanced,
            training_seeds=_SEEDS,
            previous_completion=replayed,
        )


def test_completed_plan_has_no_stage_request(tmp_path: Path) -> None:
    plan = _plan()
    cursor = initial_symbol_triplet_training_cursor(plan)
    previous_completion = None
    for stage in plan.stages:
        request = build_symbol_triplet_stage_request(
            plan,
            cursor,
            training_seeds=(0,),
            previous_completion=previous_completion,
        )
        assert request is not None
        previous_completion, cursor = commit_symbol_triplet_stage_completion(
            plan,
            cursor,
            request=request,
            checkpoint_roots=_checkpoint_roots(
                tmp_path / f"stage-{stage.stage_index:03d}",
                environment_digest=f"{stage.stage_index % 10}" * 64,
            )
            if request.training_seeds == _SEEDS
            else {
                0: _checkpoint_roots(
                    tmp_path / f"stage-{stage.stage_index:03d}",
                    environment_digest=f"{stage.stage_index % 10}" * 64,
                )[0]
            },
            completion_path=tmp_path / f"completion-{stage.stage_index:03d}.json",
            cursor_path=tmp_path / "cursor.json",
        )

    assert current_symbol_triplet_training_stage(plan, cursor) is None
    assert (
        build_symbol_triplet_stage_request(
            plan,
            cursor,
            training_seeds=(0,),
            previous_completion=previous_completion,
        )
        is None
    )
