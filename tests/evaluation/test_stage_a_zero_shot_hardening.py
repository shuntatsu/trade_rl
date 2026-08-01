from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAEvaluationObservation,
    build_stage_a_evaluation_evidence,
    build_stage_a_zero_shot_evaluation_plan,
    write_stage_a_evaluation_evidence,
    write_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    StageAValidationSelection,
    evaluate_stage_a_sealed_test,
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


def _plan(**overrides: object):
    manifest = stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("feature"),
        validation_triplet_ids=_VALIDATION_TRIPLETS,
        test_triplet_ids=_TEST_TRIPLETS,
        folds=_FOLDS,
    )
    values: dict[str, object] = {
        "symbol_disjoint_manifest_digest": manifest.symbol_disjoint_manifest_digest,
        "symbol_disjoint_triplet_manifest_digest": (
            manifest.symbol_disjoint_triplet_manifest_digest
        ),
        "evaluation_dataset_manifest_digest": manifest.digest,
        "feature_identity": manifest.feature_identity,
        "execution_identity": _digest("execution"),
        "evaluation_identity": _digest("evaluation"),
        "candidates": (_candidate("candidate-a"), _candidate("candidate-b")),
        "seeds": _SEEDS,
        "folds": _FOLDS,
        "validation_triplet_ids": _VALIDATION_TRIPLETS,
        "test_triplet_ids": _TEST_TRIPLETS,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_resamples": 2_000,
        "bootstrap_seed": 31,
        "minimum_validation_lower_bound": 0.0,
        "minimum_test_lower_bound": 0.0,
        "minimum_validation_worst_triplet_excess": 0.0,
        "minimum_test_worst_triplet_excess": 0.0,
        "minimum_validation_worst_seed_excess": 0.0,
        "minimum_test_worst_seed_excess": 0.0,
        "minimum_validation_triplet_pass_fraction": 1.0,
        "minimum_test_triplet_pass_fraction": 1.0,
    }
    values.update(overrides)
    return build_stage_a_zero_shot_evaluation_plan(**values)  # type: ignore[arg-type]


def _observation(
    *,
    plan,
    candidate_id: str,
    split: str,
    triplet_id: str,
    fold: int,
    seed: int,
    policy_growth: float,
    baseline_growth: float,
    baseline_digest: str | None = None,
    feature_identity: str | None = None,
    execution_identity: str | None = None,
    evaluation_identity: str | None = None,
) -> StageAEvaluationObservation:
    manifest = stage_a_test_manifest_for_plan(plan)
    return StageAEvaluationObservation.create(
        candidate_id=candidate_id,
        split=split,
        triplet_id=triplet_id,
        fold=fold,
        seed=seed,
        checkpoint_digest=plan.candidate(candidate_id).checkpoint_digest(seed),
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for(split, triplet_id),
        evaluation_range=manifest.range_for(split, fold),
        feature_identity=feature_identity or plan.feature_identity,
        execution_identity=execution_identity or plan.execution_identity,
        evaluation_identity=evaluation_identity or plan.evaluation_identity,
        policy_execution_evidence_digest=_digest(
            f"policy-{candidate_id}-{split}-{triplet_id}-{fold}-{seed}"
        ),
        baseline_execution_evidence_digest=(
            baseline_digest or _digest(f"baseline-{split}-{triplet_id}-{fold}-{seed}")
        ),
        policy_log_growth=policy_growth,
        baseline_log_growth=baseline_growth,
    )


def _evidence(
    *,
    plan,
    split: str,
    candidate_ids: tuple[str, ...],
    excess: dict[tuple[str, str, int, int], float],
):
    triplets = plan.triplet_ids_for(split)
    observations = tuple(
        _observation(
            plan=plan,
            candidate_id=candidate_id,
            split=split,
            triplet_id=triplet_id,
            fold=fold,
            seed=seed,
            policy_growth=0.01 + excess[(candidate_id, triplet_id, fold, seed)],
            baseline_growth=0.01,
        )
        for candidate_id in candidate_ids
        for triplet_id in triplets
        for fold in plan.folds
        for seed in plan.seeds
    )
    return build_stage_a_evaluation_evidence(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        split=split,
        candidate_ids=candidate_ids,
        observations=observations,
    )


def test_sealed_test_recomputes_validation_selection_from_validation_evidence() -> None:
    plan = _plan()
    validation_excess = {
        (candidate_id, triplet_id, fold, seed): (
            0.03 if candidate_id == "candidate-a" else 0.01
        )
        for candidate_id in plan.candidate_ids
        for triplet_id in plan.validation_triplet_ids
        for fold in plan.folds
        for seed in plan.seeds
    }
    validation = _evidence(
        plan=plan,
        split="validation",
        candidate_ids=plan.candidate_ids,
        excess=validation_excess,
    )
    actual = select_stage_a_validation_candidate(plan=plan, evidence=validation)
    forged_summaries = tuple(
        replace(
            summary,
            lower_confidence_bound=(
                0.04 if summary.candidate_id == "candidate-b" else 0.02
            ),
            digest="",
        )
        for summary in actual.candidate_summaries
    )
    forged = StageAValidationSelection(
        plan_digest=actual.plan_digest,
        validation_evidence_digest=actual.validation_evidence_digest,
        candidate_summaries=forged_summaries,
        minimum_lower_bound=actual.minimum_lower_bound,
        minimum_worst_triplet_excess=actual.minimum_worst_triplet_excess,
        minimum_worst_seed_excess=actual.minimum_worst_seed_excess,
        minimum_triplet_pass_fraction=actual.minimum_triplet_pass_fraction,
        selected_candidate_id="candidate-b",
        passed=True,
        reason="candidate_selected_by_validation_gate",
    )
    test_excess = {
        ("candidate-b", triplet_id, fold, seed): 0.02
        for triplet_id in plan.test_triplet_ids
        for fold in plan.folds
        for seed in plan.seeds
    }
    test_evidence = _evidence(
        plan=plan,
        split="test",
        candidate_ids=("candidate-b",),
        excess=test_excess,
    )

    with pytest.raises(ValueError, match="validation recomputation"):
        evaluate_stage_a_sealed_test(
            plan=plan,
            validation_evidence=validation,
            selection=forged,
            evidence=test_evidence,
        )


def test_evidence_rejects_candidate_dependent_baseline_for_the_same_cell() -> None:
    plan = _plan()
    observations: list[StageAEvaluationObservation] = []
    for candidate_id in plan.candidate_ids:
        for triplet_id in plan.validation_triplet_ids:
            for fold in plan.folds:
                for seed in plan.seeds:
                    observations.append(
                        _observation(
                            plan=plan,
                            candidate_id=candidate_id,
                            split="validation",
                            triplet_id=triplet_id,
                            fold=fold,
                            seed=seed,
                            policy_growth=0.03,
                            baseline_growth=(
                                0.01 if candidate_id == "candidate-a" else 0.02
                            ),
                            baseline_digest=_digest(
                                f"baseline-{candidate_id}-{triplet_id}-{fold}-{seed}"
                            ),
                        )
                    )

    with pytest.raises(ValueError, match="shared baseline"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            split="validation",
            observations=tuple(observations),
        )


def test_evidence_revalidates_feature_execution_and_evaluation_identities() -> None:
    plan = _plan()
    observations: list[StageAEvaluationObservation] = []
    for candidate_id in plan.candidate_ids:
        for triplet_id in plan.validation_triplet_ids:
            for fold in plan.folds:
                for seed in plan.seeds:
                    observations.append(
                        _observation(
                            plan=plan,
                            candidate_id=candidate_id,
                            split="validation",
                            triplet_id=triplet_id,
                            fold=fold,
                            seed=seed,
                            policy_growth=0.03,
                            baseline_growth=0.01,
                            execution_identity=(
                                _digest("wrong-execution")
                                if candidate_id == "candidate-a"
                                and fold == 0
                                and seed == 0
                                else plan.execution_identity
                            ),
                        )
                    )

    with pytest.raises(ValueError, match="execution identity mismatch"):
        build_stage_a_evaluation_evidence(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            split="validation",
            observations=tuple(observations),
        )


def test_validation_gate_rejects_hidden_triplet_and_seed_failures() -> None:
    plan = _plan(
        minimum_validation_lower_bound=-1.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
    )
    excess: dict[tuple[str, str, int, int], float] = {}
    for candidate_id in plan.candidate_ids:
        for triplet_index, triplet_id in enumerate(plan.validation_triplet_ids):
            for fold in plan.folds:
                for seed in plan.seeds:
                    value = 0.10
                    if candidate_id == "candidate-a" and triplet_index == 1:
                        value = -0.09
                    if candidate_id == "candidate-b" and seed == 2:
                        value = -0.09
                    excess[(candidate_id, triplet_id, fold, seed)] = value
    evidence = _evidence(
        plan=plan,
        split="validation",
        candidate_ids=plan.candidate_ids,
        excess=excess,
    )

    selection = select_stage_a_validation_candidate(plan=plan, evidence=evidence)

    assert selection.passed is False
    assert selection.selected_candidate_id is None
    assert selection.reason == "no_candidate_met_validation_gate"
    assert selection.summary("candidate-a").worst_triplet_excess_log_growth < 0.0
    assert selection.summary("candidate-b").worst_seed_excess_log_growth < 0.0


def test_bootstrap_plan_caps_resamples_and_uses_common_draw_seed() -> None:
    with pytest.raises(ValueError, match="at most"):
        _plan(bootstrap_resamples=1_000_001)

    plan = _plan()
    excess = {
        (candidate_id, triplet_id, fold, seed): 0.01 + 0.001 * fold
        for candidate_id in plan.candidate_ids
        for triplet_id in plan.validation_triplet_ids
        for fold in plan.folds
        for seed in plan.seeds
    }
    evidence = _evidence(
        plan=plan,
        split="validation",
        candidate_ids=plan.candidate_ids,
        excess=excess,
    )
    first = summarize_stage_a_candidate(
        plan=plan, evidence=evidence, candidate_id="candidate-a"
    )
    second = summarize_stage_a_candidate(
        plan=plan, evidence=evidence, candidate_id="candidate-b"
    )

    assert first.bootstrap_seed == second.bootstrap_seed


@pytest.mark.parametrize(
    "writer,value_factory",
    [
        (
            write_stage_a_zero_shot_evaluation_plan,
            lambda plan, evidence, selection, decision: plan,
        ),
        (
            write_stage_a_evaluation_evidence,
            lambda plan, evidence, selection, decision: evidence,
        ),
        (
            write_stage_a_validation_selection,
            lambda plan, evidence, selection, decision: selection,
        ),
        (
            write_stage_a_sealed_test_decision,
            lambda plan, evidence, selection, decision: decision,
        ),
    ],
)
def test_stage_a_writers_preserve_existing_file_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer,
    value_factory,
) -> None:
    plan = _plan()
    validation_excess = {
        (candidate_id, triplet_id, fold, seed): (
            0.03 if candidate_id == "candidate-a" else 0.01
        )
        for candidate_id in plan.candidate_ids
        for triplet_id in plan.validation_triplet_ids
        for fold in plan.folds
        for seed in plan.seeds
    }
    validation = _evidence(
        plan=plan,
        split="validation",
        candidate_ids=plan.candidate_ids,
        excess=validation_excess,
    )
    selection = select_stage_a_validation_candidate(plan=plan, evidence=validation)
    test_excess = {
        ("candidate-a", triplet_id, fold, seed): 0.02
        for triplet_id in plan.test_triplet_ids
        for fold in plan.folds
        for seed in plan.seeds
    }
    test_evidence = _evidence(
        plan=plan,
        split="test",
        candidate_ids=("candidate-a",),
        excess=test_excess,
    )
    decision = evaluate_stage_a_sealed_test(
        plan=plan,
        validation_evidence=validation,
        selection=selection,
        evidence=test_evidence,
    )
    value = value_factory(plan, validation, selection, decision)
    path = tmp_path / "artifact.json"
    path.write_bytes(b"previous")

    def fail_replace(
        source: os.PathLike[str] | str, destination: os.PathLike[str] | str
    ) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        writer(path, value)

    assert path.read_bytes() == b"previous"
    assert tuple(tmp_path.iterdir()) == (path,)
