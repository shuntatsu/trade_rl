from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.evaluation.replay_support import execution_episode
from tests.stage_a_helpers import stage_a_test_manifest
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.workflows.stage_a_execution_replay import (
    build_stage_a_execution_replay_artifact,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _request() -> tuple[StageAEvaluationCellRequest, str]:
    cost = ExecutionCostConfig(path_mode="conservative")
    candidate_config_digest = _digest("candidate:config")
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=candidate_config_digest,
        final_training_completion_digest=_digest("candidate:complete"),
        policy_identity=_digest("candidate:policy"),
        checkpoint_digests=(
            (7, _digest("candidate:checkpoint:7")),
            (8, _digest("candidate:checkpoint:8")),
        ),
    )
    manifest = stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbols"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplets"),
        feature_identity=_digest("features"),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        folds=(2, 3),
    )
    plan = build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=cost.execution_policy_digest,
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(7, 8),
        folds=(2, 3),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=11,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )
    request = StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=2,
        seed=7,
        candidate_id="candidate-a",
        checkpoint_digest=candidate.checkpoint_digest(7),
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for(
            "validation", plan.validation_triplet_ids[0]
        ),
        evaluation_range=manifest.range_for("validation", 2),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )
    return request, candidate_config_digest


def _source_bytes(
    tmp_path: Path,
    *,
    candidate_config_digest: str,
    evaluation_run_digest: str,
    fold: int,
    seed: int,
    actions: tuple[tuple[float, ...], ...] = ((0.4,),),
    observations: tuple[str, ...] = ("1" * 64, "2" * 64),
    equity: tuple[float, ...] = (1_000.0, 1_100.0),
) -> tuple[bytes, bytes]:
    request, _ = _request()
    events, terminal_book, terminal_order_book = execution_episode(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        cash=1_000.0,
    )
    artifact = build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=evaluation_run_digest,
        fold=fold,
        seed=seed,
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        actions=actions,
        observation_digests=observations,
        equity_curve=equity,
        order_events=events,
        terminal_book=terminal_book,
        terminal_order_book=terminal_order_book,
    )
    path = write_execution_event_artifact(tmp_path / "events.json", artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=request.dataset_id,
        cost=ExecutionCostConfig(path_mode="conservative"),
        sensitivity_path_modes=("conservative",),
        order_event_artifact_path=path,
    )
    return path.read_bytes(), canonical_json_bytes(evidence.to_mapping()) + b"\n"


def _build(
    tmp_path: Path,
    *,
    source_candidate: str | None = None,
    source_evaluation: str | None = None,
    source_fold: int | None = None,
    source_seed: int | None = None,
    source_actions: tuple[tuple[float, ...], ...] = ((0.4,),),
    source_observations: tuple[str, ...] = ("1" * 64, "2" * 64),
    source_equity: tuple[float, ...] = (1_000.0, 1_100.0),
):
    request, expected_candidate = _request()
    event_bytes, evidence_bytes = _source_bytes(
        tmp_path,
        candidate_config_digest=source_candidate or expected_candidate,
        evaluation_run_digest=source_evaluation or request.digest,
        fold=request.fold if source_fold is None else source_fold,
        seed=request.seed if source_seed is None else source_seed,
        actions=source_actions,
        observations=source_observations,
        equity=source_equity,
    )
    return build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=expected_candidate,
        actions=((0.4,),),
        observation_digests=("1" * 64, "2" * 64),
        equity_curve=(1_000.0, 1_100.0),
        event_artifact_bytes=event_bytes,
        execution_evidence_bytes=evidence_bytes,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"source_candidate": _digest("other-candidate")},
            "candidate configuration identity mismatch",
        ),
        (
            {"source_evaluation": _digest("other-evaluation-run")},
            "evaluation run identity mismatch",
        ),
        ({"source_fold": 3}, "fold identity mismatch"),
        ({"source_seed": 8}, "seed identity mismatch"),
        ({"source_actions": ((0.9,),)}, "action trace mismatch"),
        (
            {"source_observations": ("3" * 64, "4" * 64)},
            "observation trace mismatch",
        ),
        ({"source_equity": (900.0, 1_100.0)}, "equity trace mismatch"),
    ),
)
def test_stage_a_replay_rejects_lower_source_substitution(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(tmp_path, **changes)
