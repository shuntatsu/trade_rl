"""Deterministic Stage A aggregation and gate computation."""

from __future__ import annotations

from statistics import fmean

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty
from trade_rl.evaluation._stage_a_zero_shot_gate_decisions import (
    StageASealedTestDecision,
    StageAValidationSelection,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_values import (
    _BOOTSTRAP_CHUNK_SIZE,
    _TRIPLET_PASS_EXCESS_THRESHOLD,
    StageACandidateSummary,
    _summary_meets_gate,
)
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAEvaluationEvidence,
    StageAZeroShotEvaluationPlan,
)

def _bootstrap_lower_bound(
    fold_values: tuple[float, ...],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> float:
    values = np.asarray(fold_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, _BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + _BOOTSTRAP_CHUNK_SIZE, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    return float(np.quantile(means, 1.0 - confidence_level, method="lower"))


def _derived_bootstrap_seed(
    *, plan: StageAZeroShotEvaluationPlan, evidence: StageAEvaluationEvidence
) -> int:
    material = content_digest(
        {
            "bootstrap_seed": plan.bootstrap_seed,
            "evidence_digest": evidence.digest,
            "plan_digest": plan.digest,
            "schema_version": "stage_a_zero_shot_common_bootstrap_seed_v2",
        }
    )
    return int(material[:16], 16)


def summarize_stage_a_candidate(
    *,
    plan: StageAZeroShotEvaluationPlan,
    evidence: StageAEvaluationEvidence,
    candidate_id: str,
) -> StageACandidateSummary:
    evidence.validate_plan(plan)
    resolved_candidate = require_non_empty(candidate_id, field="stage_a_candidate_id")
    observations = evidence.observations_for(resolved_candidate)
    fold_values = tuple(
        (
            fold,
            fmean(item.excess_log_growth for item in observations if item.fold == fold),
        )
        for fold in plan.folds
    )
    triplet_values = tuple(
        (
            triplet_id,
            fmean(
                item.excess_log_growth
                for item in observations
                if item.triplet_id == triplet_id
            ),
        )
        for triplet_id in evidence.triplet_ids
    )
    seed_values = tuple(
        (
            seed,
            fmean(item.excess_log_growth for item in observations if item.seed == seed),
        )
        for seed in plan.seeds
    )
    expected_count = len(plan.folds) * len(plan.seeds) * len(evidence.triplet_ids)
    if len(observations) != expected_count:
        raise ValueError("Stage A candidate evidence closure mismatch")
    bootstrap_seed = _derived_bootstrap_seed(plan=plan, evidence=evidence)
    mean_excess = fmean(value for _, value in fold_values)
    lower_bound = _bootstrap_lower_bound(
        tuple(value for _, value in fold_values),
        confidence_level=plan.bootstrap_confidence_level,
        resamples=plan.bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return StageACandidateSummary(
        plan_digest=plan.digest,
        evidence_digest=evidence.digest,
        candidate_id=resolved_candidate,
        split=evidence.split,
        fold_excess_log_growth=fold_values,
        triplet_excess_log_growth=triplet_values,
        seed_excess_log_growth=seed_values,
        mean_excess_log_growth=mean_excess,
        lower_confidence_bound=lower_bound,
        worst_triplet_excess_log_growth=min(value for _, value in triplet_values),
        worst_seed_excess_log_growth=min(value for _, value in seed_values),
        triplet_pass_fraction=(
            sum(value >= _TRIPLET_PASS_EXCESS_THRESHOLD for _, value in triplet_values)
            / len(triplet_values)
        ),
        confidence_level=plan.bootstrap_confidence_level,
        bootstrap_resamples=plan.bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def select_stage_a_validation_candidate(
    *, plan: StageAZeroShotEvaluationPlan, evidence: StageAEvaluationEvidence
) -> StageAValidationSelection:
    evidence.validate_plan(plan)
    if evidence.split != "validation":
        raise ValueError("Stage A selection requires validation evidence")
    if evidence.candidate_ids != plan.candidate_ids:
        raise ValueError("Stage A validation evidence must contain every candidate")
    summaries = tuple(
        summarize_stage_a_candidate(
            plan=plan, evidence=evidence, candidate_id=candidate_id
        )
        for candidate_id in plan.candidate_ids
    )
    thresholds = plan.gate_thresholds("validation")
    eligible = tuple(
        summary
        for summary in summaries
        if _summary_meets_gate(
            summary,
            minimum_lower_bound=thresholds[0],
            minimum_worst_triplet_excess=thresholds[1],
            minimum_worst_seed_excess=thresholds[2],
            minimum_triplet_pass_fraction=thresholds[3],
        )
    )
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                -item.lower_confidence_bound,
                -item.worst_triplet_excess_log_growth,
                -item.worst_seed_excess_log_growth,
                -item.mean_excess_log_growth,
                item.candidate_id,
            ),
        ).candidate_id
        passed = True
        reason = "candidate_selected_by_validation_gate"
    else:
        selected = None
        passed = False
        reason = "no_candidate_met_validation_gate"
    return StageAValidationSelection(
        plan_digest=plan.digest,
        validation_evidence_digest=evidence.digest,
        candidate_summaries=summaries,
        minimum_lower_bound=thresholds[0],
        minimum_worst_triplet_excess=thresholds[1],
        minimum_worst_seed_excess=thresholds[2],
        minimum_triplet_pass_fraction=thresholds[3],
        selected_candidate_id=selected,
        passed=passed,
        reason=reason,
    )


def evaluate_stage_a_sealed_test(
    *,
    plan: StageAZeroShotEvaluationPlan,
    validation_evidence: StageAEvaluationEvidence,
    selection: StageAValidationSelection,
    evidence: StageAEvaluationEvidence,
) -> StageASealedTestDecision:
    expected_selection = select_stage_a_validation_candidate(
        plan=plan, evidence=validation_evidence
    )
    if selection != expected_selection:
        raise ValueError("Stage A validation selection does not match validation recomputation")
    evidence.validate_plan(plan)
    if evidence.split != "test":
        raise ValueError("Stage A sealed-test gate requires test evidence")
    if not selection.passed or selection.selected_candidate_id is None:
        raise ValueError("Stage A sealed test requires a passed validation selection")
    expected_candidates = (selection.selected_candidate_id,)
    if evidence.candidate_ids != expected_candidates:
        raise ValueError("Stage A test evidence must contain exactly the selected candidate")
    summary = summarize_stage_a_candidate(
        plan=plan, evidence=evidence, candidate_id=selection.selected_candidate_id
    )
    thresholds = plan.gate_thresholds("test")
    passed = _summary_meets_gate(
        summary,
        minimum_lower_bound=thresholds[0],
        minimum_worst_triplet_excess=thresholds[1],
        minimum_worst_seed_excess=thresholds[2],
        minimum_triplet_pass_fraction=thresholds[3],
    )
    reason = (
        "selected_candidate_met_test_gate"
        if passed
        else "selected_candidate_missed_test_gate"
    )
    return StageASealedTestDecision(
        plan_digest=plan.digest,
        validation_selection_digest=selection.digest,
        test_evidence_digest=evidence.digest,
        selected_candidate_id=selection.selected_candidate_id,
        candidate_summary=summary,
        minimum_lower_bound=thresholds[0],
        minimum_worst_triplet_excess=thresholds[1],
        minimum_worst_seed_excess=thresholds[2],
        minimum_triplet_pass_fraction=thresholds[3],
        passed=passed,
        reason=reason,
    )


