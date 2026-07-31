"""Strict JSON loaders for Stage A zero-shot evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from trade_rl.evaluation._stage_a_zero_shot_candidate import StageACandidate
from trade_rl.evaluation._stage_a_zero_shot_contract_helpers import (
    StageAEvaluationSplit,
    _integer,
    _list,
    _number,
    _object,
    _require_fields,
    _string,
)
from trade_rl.evaluation._stage_a_zero_shot_evidence import (
    StageAEvaluationEvidence,
    StageAEvaluationObservation,
)
from trade_rl.evaluation._stage_a_zero_shot_plan import StageAZeroShotEvaluationPlan


def _load_candidate(value: object, *, field: str) -> StageACandidate:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "candidate_config_digest",
            "candidate_id",
            "checkpoint_digests",
            "digest",
            "final_training_completion_digest",
            "policy_identity",
            "schema_version",
        },
        label=field,
    )
    checkpoints: list[tuple[int, str]] = []
    for index, raw in enumerate(
        _list(payload["checkpoint_digests"], field=f"{field}.checkpoint_digests")
    ):
        pair = _list(raw, field=f"{field}.checkpoint_digests[{index}]")
        if len(pair) != 2:
            raise ValueError(
                f"{field}.checkpoint_digests[{index}] must contain two values"
            )
        checkpoints.append(
            (
                _integer(pair[0], field=f"{field}.checkpoint_digests[{index}].seed"),
                _string(pair[1], field=f"{field}.checkpoint_digests[{index}].digest"),
            )
        )
    return StageACandidate(
        candidate_id=_string(payload["candidate_id"], field=f"{field}.candidate_id"),
        candidate_config_digest=_string(
            payload["candidate_config_digest"], field=f"{field}.candidate_config_digest"
        ),
        final_training_completion_digest=_string(
            payload["final_training_completion_digest"],
            field=f"{field}.final_training_completion_digest",
        ),
        policy_identity=_string(
            payload["policy_identity"], field=f"{field}.policy_identity"
        ),
        checkpoint_digests=tuple(checkpoints),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
        digest=_string(payload["digest"], field=f"{field}.digest"),
    )


def load_stage_a_zero_shot_evaluation_plan(
    path: str | Path,
) -> StageAZeroShotEvaluationPlan:
    payload = _object(
        json.loads(Path(path).read_text(encoding="utf-8")), field="stage_a_plan"
    )
    _require_fields(
        payload,
        {
            "bootstrap_confidence_level",
            "bootstrap_resamples",
            "bootstrap_seed",
            "candidates",
            "dataset_identity",
            "digest",
            "evaluation_identity",
            "execution_identity",
            "feature_identity",
            "folds",
            "minimum_test_lower_bound",
            "minimum_test_triplet_pass_fraction",
            "minimum_test_worst_seed_excess",
            "minimum_test_worst_triplet_excess",
            "minimum_validation_lower_bound",
            "minimum_validation_triplet_pass_fraction",
            "minimum_validation_worst_seed_excess",
            "minimum_validation_worst_triplet_excess",
            "schema_version",
            "seeds",
            "symbol_disjoint_manifest_digest",
            "symbol_disjoint_triplet_manifest_digest",
            "test_triplet_ids",
            "validation_triplet_ids",
        },
        label="stage_a_plan",
    )
    candidates = tuple(
        _load_candidate(value, field=f"stage_a_plan.candidates[{index}]")
        for index, value in enumerate(
            _list(payload["candidates"], field="stage_a_plan.candidates")
        )
    )
    return StageAZeroShotEvaluationPlan(
        symbol_disjoint_manifest_digest=_string(
            payload["symbol_disjoint_manifest_digest"],
            field="stage_a_plan.symbol_disjoint_manifest_digest",
        ),
        symbol_disjoint_triplet_manifest_digest=_string(
            payload["symbol_disjoint_triplet_manifest_digest"],
            field="stage_a_plan.symbol_disjoint_triplet_manifest_digest",
        ),
        dataset_identity=_string(
            payload["dataset_identity"], field="stage_a_plan.dataset_identity"
        ),
        feature_identity=_string(
            payload["feature_identity"], field="stage_a_plan.feature_identity"
        ),
        execution_identity=_string(
            payload["execution_identity"], field="stage_a_plan.execution_identity"
        ),
        evaluation_identity=_string(
            payload["evaluation_identity"], field="stage_a_plan.evaluation_identity"
        ),
        candidates=candidates,
        seeds=tuple(
            _integer(value, field=f"stage_a_plan.seeds[{index}]")
            for index, value in enumerate(
                _list(payload["seeds"], field="stage_a_plan.seeds")
            )
        ),
        folds=tuple(
            _integer(value, field=f"stage_a_plan.folds[{index}]")
            for index, value in enumerate(
                _list(payload["folds"], field="stage_a_plan.folds")
            )
        ),
        validation_triplet_ids=tuple(
            _string(value, field=f"stage_a_plan.validation_triplet_ids[{index}]")
            for index, value in enumerate(
                _list(
                    payload["validation_triplet_ids"],
                    field="stage_a_plan.validation_triplet_ids",
                )
            )
        ),
        test_triplet_ids=tuple(
            _string(value, field=f"stage_a_plan.test_triplet_ids[{index}]")
            for index, value in enumerate(
                _list(
                    payload["test_triplet_ids"], field="stage_a_plan.test_triplet_ids"
                )
            )
        ),
        bootstrap_confidence_level=_number(
            payload["bootstrap_confidence_level"],
            field="stage_a_plan.bootstrap_confidence_level",
        ),
        bootstrap_resamples=_integer(
            payload["bootstrap_resamples"], field="stage_a_plan.bootstrap_resamples"
        ),
        bootstrap_seed=_integer(
            payload["bootstrap_seed"], field="stage_a_plan.bootstrap_seed"
        ),
        minimum_validation_lower_bound=_number(
            payload["minimum_validation_lower_bound"],
            field="stage_a_plan.minimum_validation_lower_bound",
        ),
        minimum_test_lower_bound=_number(
            payload["minimum_test_lower_bound"],
            field="stage_a_plan.minimum_test_lower_bound",
        ),
        minimum_validation_worst_triplet_excess=_number(
            payload["minimum_validation_worst_triplet_excess"],
            field="stage_a_plan.minimum_validation_worst_triplet_excess",
        ),
        minimum_test_worst_triplet_excess=_number(
            payload["minimum_test_worst_triplet_excess"],
            field="stage_a_plan.minimum_test_worst_triplet_excess",
        ),
        minimum_validation_worst_seed_excess=_number(
            payload["minimum_validation_worst_seed_excess"],
            field="stage_a_plan.minimum_validation_worst_seed_excess",
        ),
        minimum_test_worst_seed_excess=_number(
            payload["minimum_test_worst_seed_excess"],
            field="stage_a_plan.minimum_test_worst_seed_excess",
        ),
        minimum_validation_triplet_pass_fraction=_number(
            payload["minimum_validation_triplet_pass_fraction"],
            field="stage_a_plan.minimum_validation_triplet_pass_fraction",
        ),
        minimum_test_triplet_pass_fraction=_number(
            payload["minimum_test_triplet_pass_fraction"],
            field="stage_a_plan.minimum_test_triplet_pass_fraction",
        ),
        schema_version=_string(
            payload["schema_version"], field="stage_a_plan.schema_version"
        ),
        digest=_string(payload["digest"], field="stage_a_plan.digest"),
    )


def _load_observation(value: object, *, field: str) -> StageAEvaluationObservation:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "baseline_execution_evidence_digest",
            "baseline_log_growth",
            "candidate_id",
            "checkpoint_digest",
            "dataset_identity",
            "digest",
            "evaluation_identity",
            "execution_identity",
            "feature_identity",
            "fold",
            "policy_execution_evidence_digest",
            "policy_log_growth",
            "schema_version",
            "seed",
            "split",
            "triplet_id",
        },
        label=field,
    )
    return StageAEvaluationObservation(
        candidate_id=_string(payload["candidate_id"], field=f"{field}.candidate_id"),
        split=cast(
            StageAEvaluationSplit, _string(payload["split"], field=f"{field}.split")
        ),
        triplet_id=_string(payload["triplet_id"], field=f"{field}.triplet_id"),
        fold=_integer(payload["fold"], field=f"{field}.fold"),
        seed=_integer(payload["seed"], field=f"{field}.seed"),
        checkpoint_digest=_string(
            payload["checkpoint_digest"], field=f"{field}.checkpoint_digest"
        ),
        dataset_identity=_string(
            payload["dataset_identity"], field=f"{field}.dataset_identity"
        ),
        feature_identity=_string(
            payload["feature_identity"], field=f"{field}.feature_identity"
        ),
        execution_identity=_string(
            payload["execution_identity"], field=f"{field}.execution_identity"
        ),
        evaluation_identity=_string(
            payload["evaluation_identity"], field=f"{field}.evaluation_identity"
        ),
        policy_execution_evidence_digest=_string(
            payload["policy_execution_evidence_digest"],
            field=f"{field}.policy_execution_evidence_digest",
        ),
        baseline_execution_evidence_digest=_string(
            payload["baseline_execution_evidence_digest"],
            field=f"{field}.baseline_execution_evidence_digest",
        ),
        policy_log_growth=_number(
            payload["policy_log_growth"], field=f"{field}.policy_log_growth"
        ),
        baseline_log_growth=_number(
            payload["baseline_log_growth"], field=f"{field}.baseline_log_growth"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
        digest=_string(payload["digest"], field=f"{field}.digest"),
    )


def load_stage_a_evaluation_evidence(
    path: str | Path, *, plan: StageAZeroShotEvaluationPlan
) -> StageAEvaluationEvidence:
    payload = _object(
        json.loads(Path(path).read_text(encoding="utf-8")), field="stage_a_evidence"
    )
    _require_fields(
        payload,
        {
            "candidate_ids",
            "digest",
            "folds",
            "observations",
            "plan_digest",
            "schema_version",
            "seeds",
            "split",
            "triplet_ids",
        },
        label="stage_a_evidence",
    )
    evidence = StageAEvaluationEvidence(
        plan_digest=_string(
            payload["plan_digest"], field="stage_a_evidence.plan_digest"
        ),
        split=cast(
            StageAEvaluationSplit,
            _string(payload["split"], field="stage_a_evidence.split"),
        ),
        candidate_ids=tuple(
            _string(value, field=f"stage_a_evidence.candidate_ids[{index}]")
            for index, value in enumerate(
                _list(payload["candidate_ids"], field="stage_a_evidence.candidate_ids")
            )
        ),
        folds=tuple(
            _integer(value, field=f"stage_a_evidence.folds[{index}]")
            for index, value in enumerate(
                _list(payload["folds"], field="stage_a_evidence.folds")
            )
        ),
        seeds=tuple(
            _integer(value, field=f"stage_a_evidence.seeds[{index}]")
            for index, value in enumerate(
                _list(payload["seeds"], field="stage_a_evidence.seeds")
            )
        ),
        triplet_ids=tuple(
            _string(value, field=f"stage_a_evidence.triplet_ids[{index}]")
            for index, value in enumerate(
                _list(payload["triplet_ids"], field="stage_a_evidence.triplet_ids")
            )
        ),
        observations=tuple(
            _load_observation(value, field=f"stage_a_evidence.observations[{index}]")
            for index, value in enumerate(
                _list(payload["observations"], field="stage_a_evidence.observations")
            )
        ),
        schema_version=_string(
            payload["schema_version"], field="stage_a_evidence.schema_version"
        ),
        digest=_string(payload["digest"], field="stage_a_evidence.digest"),
    )
    evidence.validate_plan(plan)
    return evidence
