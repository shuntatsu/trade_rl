"""Atomic writers and strict loaders for Stage A gate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation._stage_a_zero_shot_gate_compute import (
    evaluate_stage_a_sealed_test,
    select_stage_a_validation_candidate,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_decisions import (
    StageASealedTestDecision,
    StageAValidationSelection,
)
from trade_rl.evaluation._stage_a_zero_shot_gate_values import (
    StageACandidateSummary,
    _boolean,
    _integer,
    _list,
    _number,
    _object,
    _optional_string,
    _require_fields,
    _string,
)
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAEvaluationEvidence,
    StageAEvaluationSplit,
    StageAZeroShotEvaluationPlan,
)


def write_stage_a_validation_selection(
    path: str | Path, selection: StageAValidationSelection
) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(selection.to_json_dict()))


def write_stage_a_sealed_test_decision(
    path: str | Path, decision: StageASealedTestDecision
) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(decision.to_json_dict()))


def _load_int_values(value: object, *, field: str) -> tuple[tuple[int, float], ...]:
    result: list[tuple[int, float]] = []
    for index, raw in enumerate(_list(value, field=field)):
        pair = _list(raw, field=f"{field}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{field}[{index}] must contain two values")
        result.append(
            (
                _integer(pair[0], field=f"{field}[{index}].key"),
                _number(pair[1], field=f"{field}[{index}].value"),
            )
        )
    return tuple(result)


def _load_digest_values(value: object, *, field: str) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for index, raw in enumerate(_list(value, field=field)):
        pair = _list(raw, field=f"{field}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{field}[{index}] must contain two values")
        result.append(
            (
                _string(pair[0], field=f"{field}[{index}].key"),
                _number(pair[1], field=f"{field}[{index}].value"),
            )
        )
    return tuple(result)


def _load_summary(value: object, *, field: str) -> StageACandidateSummary:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "bootstrap_resamples",
            "bootstrap_seed",
            "candidate_id",
            "confidence_level",
            "digest",
            "evidence_digest",
            "fold_excess_log_growth",
            "lower_confidence_bound",
            "mean_excess_log_growth",
            "plan_digest",
            "resampling_unit",
            "schema_version",
            "seed_excess_log_growth",
            "split",
            "triplet_excess_log_growth",
            "triplet_pass_excess_threshold",
            "triplet_pass_fraction",
            "worst_seed_excess_log_growth",
            "worst_triplet_excess_log_growth",
        },
        label=field,
    )
    return StageACandidateSummary(
        plan_digest=_string(payload["plan_digest"], field=f"{field}.plan_digest"),
        evidence_digest=_string(
            payload["evidence_digest"], field=f"{field}.evidence_digest"
        ),
        candidate_id=_string(payload["candidate_id"], field=f"{field}.candidate_id"),
        split=cast(
            StageAEvaluationSplit, _string(payload["split"], field=f"{field}.split")
        ),
        fold_excess_log_growth=_load_int_values(
            payload["fold_excess_log_growth"], field=f"{field}.fold_excess_log_growth"
        ),
        triplet_excess_log_growth=_load_digest_values(
            payload["triplet_excess_log_growth"],
            field=f"{field}.triplet_excess_log_growth",
        ),
        seed_excess_log_growth=_load_int_values(
            payload["seed_excess_log_growth"], field=f"{field}.seed_excess_log_growth"
        ),
        mean_excess_log_growth=_number(
            payload["mean_excess_log_growth"], field=f"{field}.mean_excess_log_growth"
        ),
        lower_confidence_bound=_number(
            payload["lower_confidence_bound"], field=f"{field}.lower_confidence_bound"
        ),
        worst_triplet_excess_log_growth=_number(
            payload["worst_triplet_excess_log_growth"],
            field=f"{field}.worst_triplet_excess_log_growth",
        ),
        worst_seed_excess_log_growth=_number(
            payload["worst_seed_excess_log_growth"],
            field=f"{field}.worst_seed_excess_log_growth",
        ),
        triplet_pass_fraction=_number(
            payload["triplet_pass_fraction"], field=f"{field}.triplet_pass_fraction"
        ),
        confidence_level=_number(
            payload["confidence_level"], field=f"{field}.confidence_level"
        ),
        bootstrap_resamples=_integer(
            payload["bootstrap_resamples"], field=f"{field}.bootstrap_resamples"
        ),
        bootstrap_seed=_integer(
            payload["bootstrap_seed"], field=f"{field}.bootstrap_seed"
        ),
        triplet_pass_excess_threshold=_number(
            payload["triplet_pass_excess_threshold"],
            field=f"{field}.triplet_pass_excess_threshold",
        ),
        resampling_unit=_string(
            payload["resampling_unit"], field=f"{field}.resampling_unit"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
        digest=_string(payload["digest"], field=f"{field}.digest"),
    )


def load_stage_a_validation_selection(
    path: str | Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
    evidence: StageAEvaluationEvidence,
) -> StageAValidationSelection:
    payload = _object(
        json.loads(Path(path).read_text(encoding="utf-8")), field="stage_a_selection"
    )
    _require_fields(
        payload,
        {
            "candidate_summaries",
            "digest",
            "minimum_lower_bound",
            "minimum_triplet_pass_fraction",
            "minimum_worst_seed_excess",
            "minimum_worst_triplet_excess",
            "passed",
            "plan_digest",
            "reason",
            "schema_version",
            "selected_candidate_id",
            "validation_evidence_digest",
        },
        label="stage_a_selection",
    )
    selection = StageAValidationSelection(
        plan_digest=_string(
            payload["plan_digest"], field="stage_a_selection.plan_digest"
        ),
        validation_evidence_digest=_string(
            payload["validation_evidence_digest"],
            field="stage_a_selection.validation_evidence_digest",
        ),
        candidate_summaries=tuple(
            _load_summary(
                value, field=f"stage_a_selection.candidate_summaries[{index}]"
            )
            for index, value in enumerate(
                _list(
                    payload["candidate_summaries"],
                    field="stage_a_selection.candidate_summaries",
                )
            )
        ),
        minimum_lower_bound=_number(
            payload["minimum_lower_bound"],
            field="stage_a_selection.minimum_lower_bound",
        ),
        minimum_worst_triplet_excess=_number(
            payload["minimum_worst_triplet_excess"],
            field="stage_a_selection.minimum_worst_triplet_excess",
        ),
        minimum_worst_seed_excess=_number(
            payload["minimum_worst_seed_excess"],
            field="stage_a_selection.minimum_worst_seed_excess",
        ),
        minimum_triplet_pass_fraction=_number(
            payload["minimum_triplet_pass_fraction"],
            field="stage_a_selection.minimum_triplet_pass_fraction",
        ),
        selected_candidate_id=_optional_string(
            payload["selected_candidate_id"],
            field="stage_a_selection.selected_candidate_id",
        ),
        passed=_boolean(payload["passed"], field="stage_a_selection.passed"),
        reason=_string(payload["reason"], field="stage_a_selection.reason"),
        schema_version=_string(
            payload["schema_version"], field="stage_a_selection.schema_version"
        ),
        digest=_string(payload["digest"], field="stage_a_selection.digest"),
    )
    expected = select_stage_a_validation_candidate(plan=plan, evidence=evidence)
    if selection != expected:
        raise ValueError("Stage A validation selection does not match recomputation")
    return selection


def load_stage_a_sealed_test_decision(
    path: str | Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
    validation_evidence: StageAEvaluationEvidence,
    selection: StageAValidationSelection,
    evidence: StageAEvaluationEvidence,
) -> StageASealedTestDecision:
    payload = _object(
        json.loads(Path(path).read_text(encoding="utf-8")), field="stage_a_test"
    )
    _require_fields(
        payload,
        {
            "candidate_summary",
            "digest",
            "minimum_lower_bound",
            "minimum_triplet_pass_fraction",
            "minimum_worst_seed_excess",
            "minimum_worst_triplet_excess",
            "passed",
            "plan_digest",
            "reason",
            "schema_version",
            "selected_candidate_id",
            "test_evidence_digest",
            "validation_selection_digest",
        },
        label="stage_a_test",
    )
    decision = StageASealedTestDecision(
        plan_digest=_string(payload["plan_digest"], field="stage_a_test.plan_digest"),
        validation_selection_digest=_string(
            payload["validation_selection_digest"],
            field="stage_a_test.validation_selection_digest",
        ),
        test_evidence_digest=_string(
            payload["test_evidence_digest"], field="stage_a_test.test_evidence_digest"
        ),
        selected_candidate_id=_string(
            payload["selected_candidate_id"], field="stage_a_test.selected_candidate_id"
        ),
        candidate_summary=_load_summary(
            payload["candidate_summary"], field="stage_a_test.candidate_summary"
        ),
        minimum_lower_bound=_number(
            payload["minimum_lower_bound"], field="stage_a_test.minimum_lower_bound"
        ),
        minimum_worst_triplet_excess=_number(
            payload["minimum_worst_triplet_excess"],
            field="stage_a_test.minimum_worst_triplet_excess",
        ),
        minimum_worst_seed_excess=_number(
            payload["minimum_worst_seed_excess"],
            field="stage_a_test.minimum_worst_seed_excess",
        ),
        minimum_triplet_pass_fraction=_number(
            payload["minimum_triplet_pass_fraction"],
            field="stage_a_test.minimum_triplet_pass_fraction",
        ),
        passed=_boolean(payload["passed"], field="stage_a_test.passed"),
        reason=_string(payload["reason"], field="stage_a_test.reason"),
        schema_version=_string(
            payload["schema_version"], field="stage_a_test.schema_version"
        ),
        digest=_string(payload["digest"], field="stage_a_test.digest"),
    )
    expected = evaluate_stage_a_sealed_test(
        plan=plan,
        validation_evidence=validation_evidence,
        selection=selection,
        evidence=evidence,
    )
    if decision != expected:
        raise ValueError("Stage A sealed-test decision does not match recomputation")
    return decision
