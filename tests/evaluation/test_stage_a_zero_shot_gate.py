from __future__ import annotations

from pathlib import Path

import pytest

from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAEvaluationObservation,
    build_stage_a_evaluation_evidence,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    StageASealedTestDecision,
    StageAValidationSelection,
    evaluate_stage_a_sealed_test,
    load_stage_a_sealed_test_decision,
    load_stage_a_validation_selection,
    select_stage_a_validation_candidate,
    summarize_stage_a_candidate,
    write_stage_a_sealed_test_decision,
    write_stage_a_validation_selection,
)

_SEEDS = (0, 1, 2)
_FOLDS = (0, 1)
_VALIDATION_TRIPLETS = (
    content_digest({"triplet": "validation-0"}),
    content_digest({"triplet": "validation-1"}),
)
_TEST_TRIPLETS = (content_digest({"triplet": "test-0"}),)


def _digest(label: str) -> str:
    return content_digest({"label": label})


def _candidate(candidate_id: str) -> StageACandidate:
    return StageACandidate.create(
        candidate_id=candidate_id,
        candidate_config_digest=_digest(f"{candidate_id}-config"),
        final_training_completion_digest=_digest(f"{candidate_id}-completion"),
        policy_identity=_digest(f"{candidate_id}-policy"),
        checkpoint_digests=tuple(
            (seed, _digest(f"{candidate_id}-checkpoint-{seed}")) for seed in _SEEDS
        ),
    )


def _manifest():
    return stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("feature"),
        validation_triplet_ids=_VALIDATION_TRIPLETS,
        test_triplet_ids=_TEST_TRIPLETS,
        folds=_FOLDS,
    )


def _plan(*, validation_threshold: float = 0.005, test_threshold: float = 0.01):
    manifest = _manifest()
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=(_candidate("candidate-a"), _candidate("candidate-b")),
        seeds=_SEEDS,
        folds=_FOLDS,
        validation_triplet_ids=_VALIDATION_TRIPLETS,
        test_triplet_ids=_TEST_TRIPLETS,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
        bootstrap_seed=31,
        minimum_validation_lower_bound=validation_threshold,
        minimum_test_lower_bound=test_threshold,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _evidence(
    *,
    split: str,
    excess_by_candidate_fold: dict[str, tuple[float, float]],
    candidate_ids: tuple[str, ...] = ("candidate-a", "candidate-b"),
):
    plan = _plan()
    manifest = stage_a_test_manifest_for_plan(plan)
    triplets = (
        plan.validation_triplet_ids if split == "validation" else plan.test_triplet_ids
    )
    observations: list[StageAEvaluationObservation] = []
    for candidate_id in candidate_ids:
        candidate = plan.candidate(candidate_id)
        checkpoints = dict(candidate.checkpoint_digests)
        for triplet_index, triplet_id in enumerate(triplets):
            for fold in plan.folds:
                for seed in plan.seeds:
                    baseline = 0.01 + 0.0001 * seed
                    observations.append(
                        StageAEvaluationObservation.create(
                            candidate_id=candidate_id,
                            split=split,
                            triplet_id=triplet_id,
                            fold=fold,
                            seed=seed,
                            checkpoint_digest=checkpoints[seed],
                            evaluation_dataset_manifest_digest=manifest.digest,
                            dataset_id=manifest.dataset_id_for(split, triplet_id),
                            evaluation_range=manifest.range_for(split, fold),
                            feature_identity=plan.feature_identity,
                            execution_identity=plan.execution_identity,
                            evaluation_identity=plan.evaluation_identity,
                            policy_execution_evidence_digest=_digest(
                                f"policy-execution-{split}-{candidate_id}-{triplet_index}-{fold}-{seed}"
                            ),
                            baseline_execution_evidence_digest=_digest(
                                f"baseline-execution-{split}-{triplet_index}-{fold}-{seed}"
                            ),
                            policy_log_growth=(
                                baseline + excess_by_candidate_fold[candidate_id][fold]
                            ),
                            baseline_log_growth=baseline,
                        )
                    )
    return plan, build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split=split,
        candidate_ids=candidate_ids,
        observations=tuple(observations),
    )


def test_validation_selection_uses_fold_units_and_is_deterministic(
    tmp_path: Path,
) -> None:
    plan, evidence = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (-0.01, -0.005),
        },
    )

    summary = summarize_stage_a_candidate(
        plan=plan,
        evidence=evidence,
        candidate_id="candidate-a",
    )
    first = select_stage_a_validation_candidate(plan=plan, evidence=evidence)
    repeated = select_stage_a_validation_candidate(plan=plan, evidence=evidence)

    assert tuple(fold for fold, _ in summary.fold_excess_log_growth) == (0, 1)
    assert tuple(value for _, value in summary.fold_excess_log_growth) == pytest.approx(
        (0.02, 0.03)
    )
    assert summary.mean_excess_log_growth == pytest.approx(0.025)
    assert summary.lower_confidence_bound >= 0.019
    assert first == repeated
    assert first.digest == repeated.digest
    assert first.passed is True
    assert first.selected_candidate_id == "candidate-a"

    path = write_stage_a_validation_selection(tmp_path / "selection.json", first)
    assert (
        load_stage_a_validation_selection(path, plan=plan, evidence=evidence) == first
    )


def test_validation_selection_fails_when_no_candidate_reaches_threshold() -> None:
    plan = _plan(validation_threshold=0.05)
    _, raw_evidence = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (0.01, 0.015),
        },
    )
    evidence = build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split="validation",
        observations=raw_evidence.observations,
    )

    selection = select_stage_a_validation_candidate(plan=plan, evidence=evidence)

    assert selection.passed is False
    assert selection.selected_candidate_id is None
    assert selection.reason == "no_candidate_met_validation_gate"


def test_validation_selector_rejects_test_evidence() -> None:
    plan, evidence = _evidence(
        split="test",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (0.01, 0.02),
        },
    )

    with pytest.raises(ValueError, match="validation evidence"):
        select_stage_a_validation_candidate(plan=plan, evidence=evidence)


def test_sealed_test_requires_exactly_the_selected_candidate() -> None:
    plan, validation = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (-0.01, -0.005),
        },
    )
    selection = select_stage_a_validation_candidate(plan=plan, evidence=validation)
    _, all_candidates_test = _evidence(
        split="test",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.025),
            "candidate-b": (0.03, 0.035),
        },
    )

    with pytest.raises(ValueError, match="exactly the selected candidate"):
        evaluate_stage_a_sealed_test(
            plan=plan,
            validation_evidence=validation,
            selection=selection,
            evidence=all_candidates_test,
        )
    with pytest.raises(ValueError, match="test evidence"):
        evaluate_stage_a_sealed_test(
            plan=plan,
            validation_evidence=validation,
            selection=selection,
            evidence=validation,
        )


def test_sealed_test_passes_or_fails_without_changing_the_selection(
    tmp_path: Path,
) -> None:
    plan, validation = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (-0.01, -0.005),
        },
    )
    selection = select_stage_a_validation_candidate(plan=plan, evidence=validation)
    _, passing_test = _evidence(
        split="test",
        candidate_ids=("candidate-a",),
        excess_by_candidate_fold={"candidate-a": (0.015, 0.02)},
    )
    passed = evaluate_stage_a_sealed_test(
        plan=plan,
        validation_evidence=validation,
        selection=selection,
        evidence=passing_test,
    )

    assert passed.selected_candidate_id == "candidate-a"
    assert passed.passed is True
    assert passed.reason == "selected_candidate_met_test_gate"
    path = write_stage_a_sealed_test_decision(tmp_path / "decision.json", passed)
    assert (
        load_stage_a_sealed_test_decision(
            path,
            plan=plan,
            validation_evidence=validation,
            selection=selection,
            evidence=passing_test,
        )
        == passed
    )

    _, failing_test = _evidence(
        split="test",
        candidate_ids=("candidate-a",),
        excess_by_candidate_fold={"candidate-a": (-0.02, -0.01)},
    )
    failed = evaluate_stage_a_sealed_test(
        plan=plan,
        validation_evidence=validation,
        selection=selection,
        evidence=failing_test,
    )

    assert failed.selected_candidate_id == selection.selected_candidate_id
    assert failed.passed is False
    assert failed.reason == "selected_candidate_missed_test_gate"


def test_validation_selection_rejects_a_forged_winner() -> None:
    plan, evidence = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (0.01, 0.015),
        },
    )
    valid = select_stage_a_validation_candidate(plan=plan, evidence=evidence)

    with pytest.raises(ValueError, match="deterministic winner"):
        StageAValidationSelection(
            plan_digest=valid.plan_digest,
            validation_evidence_digest=valid.validation_evidence_digest,
            candidate_summaries=valid.candidate_summaries,
            minimum_lower_bound=valid.minimum_lower_bound,
            minimum_worst_triplet_excess=valid.minimum_worst_triplet_excess,
            minimum_worst_seed_excess=valid.minimum_worst_seed_excess,
            minimum_triplet_pass_fraction=valid.minimum_triplet_pass_fraction,
            selected_candidate_id="candidate-b",
            passed=True,
            reason="candidate_selected_by_validation_gate",
        )


def test_validation_selection_rejects_a_forged_failure() -> None:
    plan, evidence = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (-0.01, -0.005),
        },
    )
    valid = select_stage_a_validation_candidate(plan=plan, evidence=evidence)

    with pytest.raises(ValueError, match="eligible candidate"):
        StageAValidationSelection(
            plan_digest=valid.plan_digest,
            validation_evidence_digest=valid.validation_evidence_digest,
            candidate_summaries=valid.candidate_summaries,
            minimum_lower_bound=valid.minimum_lower_bound,
            minimum_worst_triplet_excess=valid.minimum_worst_triplet_excess,
            minimum_worst_seed_excess=valid.minimum_worst_seed_excess,
            minimum_triplet_pass_fraction=valid.minimum_triplet_pass_fraction,
            selected_candidate_id=None,
            passed=False,
            reason="no_candidate_met_validation_gate",
        )


def test_sealed_test_decision_rejects_a_forged_outcome() -> None:
    plan, validation = _evidence(
        split="validation",
        excess_by_candidate_fold={
            "candidate-a": (0.02, 0.03),
            "candidate-b": (-0.01, -0.005),
        },
    )
    selection = select_stage_a_validation_candidate(plan=plan, evidence=validation)
    _, test_evidence = _evidence(
        split="test",
        candidate_ids=("candidate-a",),
        excess_by_candidate_fold={"candidate-a": (0.015, 0.02)},
    )
    valid = evaluate_stage_a_sealed_test(
        plan=plan,
        validation_evidence=validation,
        selection=selection,
        evidence=test_evidence,
    )

    with pytest.raises(ValueError, match="outcome mismatch"):
        StageASealedTestDecision(
            plan_digest=valid.plan_digest,
            validation_selection_digest=valid.validation_selection_digest,
            test_evidence_digest=valid.test_evidence_digest,
            selected_candidate_id=valid.selected_candidate_id,
            candidate_summary=valid.candidate_summary,
            minimum_lower_bound=valid.minimum_lower_bound,
            minimum_worst_triplet_excess=valid.minimum_worst_triplet_excess,
            minimum_worst_seed_excess=valid.minimum_worst_seed_excess,
            minimum_triplet_pass_fraction=valid.minimum_triplet_pass_fraction,
            passed=False,
            reason="selected_candidate_missed_test_gate",
        )
