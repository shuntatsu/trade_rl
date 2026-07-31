from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.orders import OrderBookState, OrderEvent, OrderStatus
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_production_evaluator import (
    ArtifactBackedStageAEvaluationCellEvaluator,
)
from trade_rl.workflows.stage_a_zero_shot_runner import (
    StageAZeroShotEvaluationOrchestrator,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
    StageATestFoldRange,
    StageATestSchedule,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_BASELINE_CONFIG = _digest("baseline:config")


def _plan() -> StageAZeroShotEvaluationPlan:
    seeds = (0, 1)
    candidates = tuple(
        StageACandidate.create(
            candidate_id=candidate_id,
            candidate_config_digest=_digest(f"{candidate_id}:config"),
            final_training_completion_digest=_digest(f"{candidate_id}:complete"),
            policy_identity=_digest(f"{candidate_id}:policy"),
            checkpoint_digests=tuple(
                (seed, _digest(f"{candidate_id}:checkpoint:{seed}"))
                for seed in seeds
            ),
        )
        for candidate_id in ("candidate-a", "candidate-b")
    )
    cost = ExecutionCostConfig(path_mode="conservative")
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
        execution_identity=cost.execution_policy_digest,
        evaluation_identity=_digest("evaluation"),
        candidates=candidates,
        seeds=seeds,
        folds=(0, 1),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=0.01,
        minimum_test_lower_bound=0.01,
        minimum_validation_worst_triplet_excess=0.01,
        minimum_test_worst_triplet_excess=0.01,
        minimum_validation_worst_seed_excess=0.01,
        minimum_test_worst_seed_excess=0.01,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _event(plan: StageAZeroShotEvaluationPlan) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=0,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id=plan.dataset_identity,
        execution_policy_digest=plan.execution_identity,
        symbol_index=0,
        event_type="filled",
        processing_index=1,
        timestamp_ns=1,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=OrderStatus.FILLED,
        requested_quantity=1.0,
        remaining_quantity=0.0,
        filled_quantity=1.0,
        execution_price=100.0,
        filled_notional=100.0,
        capacity_before=10.0,
        capacity_after=9.0,
        participation_rate=0.1,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=None,
        path_mode="conservative",
        path_points=(100.0, 101.0, 99.0, 100.5),
    )


def _promotion_sources(
    root: Path,
    plan: StageAZeroShotEvaluationPlan,
    *,
    terminal_equity: float,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    event_artifact = build_execution_event_artifact(
        dataset_id=plan.dataset_identity,
        execution_policy_digest=plan.execution_identity,
        order_events=(_event(plan),),
        terminal_book=BookState(
            quantities=np.array((1.0,), dtype=np.float64),
            cash=terminal_equity - 100.0,
            mark_prices=np.array((100.0,), dtype=np.float64),
            peak_value=max(1_000.0, terminal_equity),
        ),
        terminal_order_book=OrderBookState.empty(),
    )
    event_path = write_execution_event_artifact(root / "events.json", event_artifact)
    evidence = execution_evidence_from_cost(
        dataset_id=plan.dataset_identity,
        cost=ExecutionCostConfig(path_mode="conservative"),
        sensitivity_path_modes=("conservative",),
        order_event_artifact_path=event_path,
    )
    evidence_path = root / "evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence.to_mapping()) + b"\n")
    return event_path, evidence_path


def _request(
    plan: StageAZeroShotEvaluationPlan,
    *,
    fold: int,
    seed: int,
    candidate_id: str | None,
) -> StageAEvaluationCellRequest:
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=fold,
        seed=seed,
        candidate_id=candidate_id,
        checkpoint_digest=(
            None
            if candidate_id is None
            else plan.candidate(candidate_id).checkpoint_digest(seed)
        ),
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _publish_validation_cells(
    *,
    root: Path,
    plan: StageAZeroShotEvaluationPlan,
    store: StageAExecutionPromotionStore,
) -> None:
    final_equity = {None: 1_000.0, "candidate-a": 1_120.0, "candidate-b": 1_060.0}
    for fold in plan.folds:
        for seed in plan.seeds:
            for candidate_id in (None, *plan.candidate_ids):
                request = _request(
                    plan,
                    fold=fold,
                    seed=seed,
                    candidate_id=candidate_id,
                )
                config_digest = (
                    _BASELINE_CONFIG
                    if candidate_id is None
                    else plan.candidate(candidate_id).candidate_config_digest
                )
                terminal_equity = final_equity[candidate_id]
                event_path, evidence_path = _promotion_sources(
                    root / "source" / request.digest,
                    plan,
                    terminal_equity=terminal_equity,
                )
                store.publish(
                    request=request,
                    candidate_config_digest=config_digest,
                    actions=((0.4,),),
                    observation_digests=(
                        _digest(f"observation:{request.digest}:0"),
                        _digest(f"observation:{request.digest}:1"),
                    ),
                    equity_curve=(1_000.0, terminal_equity),
                    event_artifact_path=event_path,
                    execution_evidence_path=evidence_path,
                )


def test_a6a_validation_consumes_verified_cells_with_shared_baseline(
    tmp_path: Path,
) -> None:
    plan = _plan()
    store = StageAExecutionPromotionStore(tmp_path / "store")
    _publish_validation_cells(root=tmp_path, plan=plan, store=store)
    evaluator = ArtifactBackedStageAEvaluationCellEvaluator(
        plan=plan,
        store=store,
        baseline_candidate_config_digest=_BASELINE_CONFIG,
    )
    schedule = StageATestSchedule(
        plan_digest=plan.digest,
        evaluation_identity=plan.evaluation_identity,
        fold_ranges=tuple(
            StageATestFoldRange(
                fold=fold,
                test_range=IndexRange(100 + fold * 20, 120 + fold * 20),
            )
            for fold in plan.folds
        ),
    )
    orchestrator = StageAZeroShotEvaluationOrchestrator(
        plan=plan,
        evaluator=evaluator,
        test_schedule=schedule,
    )

    run = orchestrator.evaluate_validation()

    assert run.selection.passed
    assert run.selection.selected_candidate_id == "candidate-a"
    assert len(run.evidence.observations) == 8
    baselines: dict[tuple[str, int, int], set[str]] = {}
    for observation in run.evidence.observations:
        baselines.setdefault(observation.baseline_key, set()).add(
            observation.baseline_execution_evidence_digest
        )
        assert observation.policy_execution_evidence_digest != (
            observation.baseline_execution_evidence_digest
        )
    assert all(len(digests) == 1 for digests in baselines.values())
    assert len({next(iter(digests)) for digests in baselines.values()}) == 4
