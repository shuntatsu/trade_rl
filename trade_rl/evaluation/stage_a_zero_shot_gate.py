"""Fold-level Stage A validation selection and sealed-test decisions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Final, cast

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAEvaluationEvidence,
    StageAEvaluationSplit,
    StageAZeroShotEvaluationPlan,
)

STAGE_A_CANDIDATE_SUMMARY_SCHEMA: Final = "stage_a_zero_shot_candidate_summary_v1"
STAGE_A_VALIDATION_SELECTION_SCHEMA: Final = "stage_a_zero_shot_validation_selection_v1"
STAGE_A_SEALED_TEST_DECISION_SCHEMA: Final = "stage_a_zero_shot_sealed_test_decision_v1"
_RESAMPLING_UNIT: Final = "fold"


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_fields(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} field closure mismatch")


@dataclass(frozen=True, slots=True)
class StageACandidateSummary:
    """Fold-aggregated paired excess growth and its one-sided lower bound."""

    plan_digest: str
    evidence_digest: str
    candidate_id: str
    split: StageAEvaluationSplit
    fold_excess_log_growth: tuple[tuple[int, float], ...]
    mean_excess_log_growth: float
    lower_confidence_bound: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    resampling_unit: str = _RESAMPLING_UNIT
    schema_version: str = STAGE_A_CANDIDATE_SUMMARY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_CANDIDATE_SUMMARY_SCHEMA:
            raise ValueError("unsupported Stage A candidate summary schema")
        require_sha256(self.plan_digest, field="stage_a_summary.plan_digest")
        require_sha256(self.evidence_digest, field="stage_a_summary.evidence_digest")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_summary.candidate_id"
        )
        if self.split not in {"validation", "test"}:
            raise ValueError("Stage A candidate summary split is invalid")
        if self.resampling_unit != _RESAMPLING_UNIT:
            raise ValueError("Stage A candidate summary must resample folds")
        if len(self.fold_excess_log_growth) < 2:
            raise ValueError("Stage A candidate summary requires at least two folds")
        normalized: list[tuple[int, float]] = []
        seen: set[int] = set()
        for fold, value in self.fold_excess_log_growth:
            resolved_fold = _non_negative_int(fold, field="stage_a_summary.fold")
            if resolved_fold in seen:
                raise ValueError("Stage A candidate summary folds must be unique")
            seen.add(resolved_fold)
            normalized.append(
                (
                    resolved_fold,
                    _finite(value, field="stage_a_summary.fold_excess_log_growth"),
                )
            )
        fold_values = tuple(sorted(normalized))
        mean_value = _finite(
            self.mean_excess_log_growth,
            field="stage_a_summary.mean_excess_log_growth",
        )
        expected_mean = fmean(value for _, value in fold_values)
        if not math.isclose(mean_value, expected_mean, rel_tol=1e-15, abs_tol=1e-15):
            raise ValueError("Stage A candidate summary mean mismatch")
        lower_bound = _finite(
            self.lower_confidence_bound,
            field="stage_a_summary.lower_confidence_bound",
        )
        confidence = _finite(
            self.confidence_level, field="stage_a_summary.confidence_level"
        )
        if not 0.5 < confidence < 1.0:
            raise ValueError("Stage A candidate summary confidence is invalid")
        resamples = _positive_int(
            self.bootstrap_resamples, field="stage_a_summary.bootstrap_resamples"
        )
        bootstrap_seed = _non_negative_int(
            self.bootstrap_seed, field="stage_a_summary.bootstrap_seed"
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "fold_excess_log_growth", fold_values)
        object.__setattr__(self, "mean_excess_log_growth", mean_value)
        object.__setattr__(self, "lower_confidence_bound", lower_bound)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A candidate summary digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "candidate_id": self.candidate_id,
            "confidence_level": self.confidence_level,
            "evidence_digest": self.evidence_digest,
            "fold_excess_log_growth": self.fold_excess_log_growth,
            "lower_confidence_bound": self.lower_confidence_bound,
            "mean_excess_log_growth": self.mean_excess_log_growth,
            "plan_digest": self.plan_digest,
            "resampling_unit": self.resampling_unit,
            "schema_version": self.schema_version,
            "split": self.split,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageAValidationSelection:
    """Validation-only deterministic candidate selection artifact."""

    plan_digest: str
    validation_evidence_digest: str
    candidate_summaries: tuple[StageACandidateSummary, ...]
    minimum_lower_bound: float
    selected_candidate_id: str | None
    passed: bool
    reason: str
    schema_version: str = STAGE_A_VALIDATION_SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_VALIDATION_SELECTION_SCHEMA:
            raise ValueError("unsupported Stage A validation selection schema")
        require_sha256(self.plan_digest, field="stage_a_selection.plan_digest")
        require_sha256(
            self.validation_evidence_digest,
            field="stage_a_selection.validation_evidence_digest",
        )
        if not self.candidate_summaries:
            raise ValueError("Stage A validation summaries must not be empty")
        summaries = tuple(
            sorted(self.candidate_summaries, key=lambda item: item.candidate_id)
        )
        candidate_ids = tuple(item.candidate_id for item in summaries)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Stage A validation summary candidates must be unique")
        if any(item.split != "validation" for item in summaries):
            raise ValueError("Stage A selection requires validation summaries")
        if any(
            item.plan_digest != self.plan_digest
            or item.evidence_digest != self.validation_evidence_digest
            for item in summaries
        ):
            raise ValueError("Stage A validation summary identity mismatch")
        threshold = _finite(
            self.minimum_lower_bound,
            field="stage_a_selection.minimum_lower_bound",
        )
        if not isinstance(self.passed, bool):
            raise ValueError("Stage A validation passed must be boolean")
        reason = require_non_empty(self.reason, field="stage_a_selection.reason")
        selected = self.selected_candidate_id
        eligible = tuple(
            item for item in summaries if item.lower_confidence_bound >= threshold
        )
        if eligible:
            if not self.passed:
                raise ValueError(
                    "Stage A validation selection cannot fail with an eligible candidate"
                )
            expected_summary = min(
                eligible,
                key=lambda item: (
                    -item.lower_confidence_bound,
                    -item.mean_excess_log_growth,
                    item.candidate_id,
                ),
            )
            if selected != expected_summary.candidate_id:
                raise ValueError(
                    "Stage A validation selection must use the deterministic winner"
                )
            if reason != "candidate_selected_by_validation_lower_bound":
                raise ValueError("Stage A validation selection reason mismatch")
        else:
            if self.passed or selected is not None:
                raise ValueError(
                    "Stage A validation selection cannot pass without an eligible candidate"
                )
            if reason != "no_candidate_met_validation_lower_bound":
                raise ValueError("Stage A validation selection reason mismatch")
        object.__setattr__(self, "candidate_summaries", summaries)
        object.__setattr__(self, "minimum_lower_bound", threshold)
        object.__setattr__(self, "reason", reason)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A validation selection digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def summary(self, candidate_id: str) -> StageACandidateSummary:
        resolved = require_non_empty(candidate_id, field="stage_a_candidate_id")
        for summary in self.candidate_summaries:
            if summary.candidate_id == resolved:
                return summary
        raise ValueError("Stage A selection candidate summary is missing")

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_summaries": tuple(
                item.to_json_dict() for item in self.candidate_summaries
            ),
            "minimum_lower_bound": self.minimum_lower_bound,
            "passed": self.passed,
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "validation_evidence_digest": self.validation_evidence_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageASealedTestDecision:
    """Final unseen-symbol gate for the already selected candidate."""

    plan_digest: str
    validation_selection_digest: str
    test_evidence_digest: str
    selected_candidate_id: str
    candidate_summary: StageACandidateSummary
    minimum_lower_bound: float
    passed: bool
    reason: str
    schema_version: str = STAGE_A_SEALED_TEST_DECISION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_SEALED_TEST_DECISION_SCHEMA:
            raise ValueError("unsupported Stage A sealed-test decision schema")
        require_sha256(self.plan_digest, field="stage_a_test.plan_digest")
        require_sha256(
            self.validation_selection_digest,
            field="stage_a_test.validation_selection_digest",
        )
        require_sha256(
            self.test_evidence_digest, field="stage_a_test.test_evidence_digest"
        )
        candidate_id = require_non_empty(
            self.selected_candidate_id, field="stage_a_test.selected_candidate_id"
        )
        if self.candidate_summary.split != "test":
            raise ValueError("Stage A sealed-test decision requires a test summary")
        if (
            self.candidate_summary.plan_digest != self.plan_digest
            or self.candidate_summary.evidence_digest != self.test_evidence_digest
            or self.candidate_summary.candidate_id != candidate_id
        ):
            raise ValueError("Stage A sealed-test summary identity mismatch")
        threshold = _finite(
            self.minimum_lower_bound, field="stage_a_test.minimum_lower_bound"
        )
        if not isinstance(self.passed, bool):
            raise ValueError("Stage A sealed-test passed must be boolean")
        reason = require_non_empty(self.reason, field="stage_a_test.reason")
        expected_passed = self.candidate_summary.lower_confidence_bound >= threshold
        expected_reason = (
            "selected_candidate_met_test_lower_bound"
            if expected_passed
            else "selected_candidate_missed_test_lower_bound"
        )
        if self.passed != expected_passed or reason != expected_reason:
            raise ValueError("Stage A sealed-test outcome mismatch")
        object.__setattr__(self, "selected_candidate_id", candidate_id)
        object.__setattr__(self, "minimum_lower_bound", threshold)
        object.__setattr__(self, "reason", reason)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A sealed-test decision digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_summary": self.candidate_summary.to_json_dict(),
            "minimum_lower_bound": self.minimum_lower_bound,
            "passed": self.passed,
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "test_evidence_digest": self.test_evidence_digest,
            "validation_selection_digest": self.validation_selection_digest,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def _bootstrap_lower_bound(
    fold_values: tuple[float, ...],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> float:
    values = np.asarray(fold_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1, dtype=np.float64)
    return float(np.quantile(means, 1.0 - confidence_level, method="lower"))


def _derived_bootstrap_seed(
    *,
    plan: StageAZeroShotEvaluationPlan,
    evidence: StageAEvaluationEvidence,
    candidate_id: str,
) -> int:
    material = content_digest(
        {
            "bootstrap_seed": plan.bootstrap_seed,
            "candidate_id": candidate_id,
            "evidence_digest": evidence.digest,
            "plan_digest": plan.digest,
            "schema_version": "stage_a_zero_shot_bootstrap_seed_v1",
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
    fold_values: list[tuple[int, float]] = []
    for fold in plan.folds:
        values = tuple(
            item.excess_log_growth for item in observations if item.fold == fold
        )
        expected_count = len(plan.seeds) * len(evidence.triplet_ids)
        if len(values) != expected_count:
            raise ValueError("Stage A candidate fold evidence closure mismatch")
        fold_values.append((fold, fmean(values)))
    bootstrap_seed = _derived_bootstrap_seed(
        plan=plan,
        evidence=evidence,
        candidate_id=resolved_candidate,
    )
    resolved_fold_values = tuple(fold_values)
    mean_excess = fmean(value for _, value in resolved_fold_values)
    lower_bound = _bootstrap_lower_bound(
        tuple(value for _, value in resolved_fold_values),
        confidence_level=plan.bootstrap_confidence_level,
        resamples=plan.bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return StageACandidateSummary(
        plan_digest=plan.digest,
        evidence_digest=evidence.digest,
        candidate_id=resolved_candidate,
        split=evidence.split,
        fold_excess_log_growth=resolved_fold_values,
        mean_excess_log_growth=mean_excess,
        lower_confidence_bound=lower_bound,
        confidence_level=plan.bootstrap_confidence_level,
        bootstrap_resamples=plan.bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def select_stage_a_validation_candidate(
    *,
    plan: StageAZeroShotEvaluationPlan,
    evidence: StageAEvaluationEvidence,
) -> StageAValidationSelection:
    evidence.validate_plan(plan)
    if evidence.split != "validation":
        raise ValueError("Stage A selection requires validation evidence")
    if evidence.candidate_ids != plan.candidate_ids:
        raise ValueError("Stage A validation evidence must contain every candidate")
    summaries = tuple(
        summarize_stage_a_candidate(
            plan=plan,
            evidence=evidence,
            candidate_id=candidate_id,
        )
        for candidate_id in plan.candidate_ids
    )
    eligible = tuple(
        item
        for item in summaries
        if item.lower_confidence_bound >= plan.minimum_validation_lower_bound
    )
    if not eligible:
        selected = None
        passed = False
        reason = "no_candidate_met_validation_lower_bound"
    else:
        selected_summary = min(
            eligible,
            key=lambda item: (
                -item.lower_confidence_bound,
                -item.mean_excess_log_growth,
                item.candidate_id,
            ),
        )
        selected = selected_summary.candidate_id
        passed = True
        reason = "candidate_selected_by_validation_lower_bound"
    return StageAValidationSelection(
        plan_digest=plan.digest,
        validation_evidence_digest=evidence.digest,
        candidate_summaries=summaries,
        minimum_lower_bound=plan.minimum_validation_lower_bound,
        selected_candidate_id=selected,
        passed=passed,
        reason=reason,
    )


def evaluate_stage_a_sealed_test(
    *,
    plan: StageAZeroShotEvaluationPlan,
    selection: StageAValidationSelection,
    evidence: StageAEvaluationEvidence,
) -> StageASealedTestDecision:
    evidence.validate_plan(plan)
    if evidence.split != "test":
        raise ValueError("Stage A sealed-test gate requires test evidence")
    if selection.plan_digest != plan.digest:
        raise ValueError("Stage A validation selection plan mismatch")
    if not selection.passed or selection.selected_candidate_id is None:
        raise ValueError("Stage A sealed test requires a passed validation selection")
    expected_candidates = (selection.selected_candidate_id,)
    if evidence.candidate_ids != expected_candidates:
        raise ValueError(
            "Stage A test evidence must contain exactly the selected candidate"
        )
    summary = summarize_stage_a_candidate(
        plan=plan,
        evidence=evidence,
        candidate_id=selection.selected_candidate_id,
    )
    passed = summary.lower_confidence_bound >= plan.minimum_test_lower_bound
    reason = (
        "selected_candidate_met_test_lower_bound"
        if passed
        else "selected_candidate_missed_test_lower_bound"
    )
    return StageASealedTestDecision(
        plan_digest=plan.digest,
        validation_selection_digest=selection.digest,
        test_evidence_digest=evidence.digest,
        selected_candidate_id=selection.selected_candidate_id,
        candidate_summary=summary,
        minimum_lower_bound=plan.minimum_test_lower_bound,
        passed=passed,
        reason=reason,
    )


def write_stage_a_validation_selection(
    path: str | Path, selection: StageAValidationSelection
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(selection.to_json_dict()))
    return output


def write_stage_a_sealed_test_decision(
    path: str | Path, decision: StageASealedTestDecision
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(decision.to_json_dict()))
    return output


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
            "split",
        },
        label=field,
    )
    raw_folds = _list(
        payload["fold_excess_log_growth"], field=f"{field}.fold_excess_log_growth"
    )
    fold_values: list[tuple[int, float]] = []
    for index, raw in enumerate(raw_folds):
        pair = _list(raw, field=f"{field}.fold_excess_log_growth[{index}]")
        if len(pair) != 2:
            raise ValueError(
                f"{field}.fold_excess_log_growth[{index}] must contain two values"
            )
        fold_values.append(
            (
                _integer(
                    pair[0], field=f"{field}.fold_excess_log_growth[{index}].fold"
                ),
                _number(
                    pair[1], field=f"{field}.fold_excess_log_growth[{index}].value"
                ),
            )
        )
    return StageACandidateSummary(
        plan_digest=_string(payload["plan_digest"], field=f"{field}.plan_digest"),
        evidence_digest=_string(
            payload["evidence_digest"], field=f"{field}.evidence_digest"
        ),
        candidate_id=_string(payload["candidate_id"], field=f"{field}.candidate_id"),
        split=cast(
            StageAEvaluationSplit,
            _string(payload["split"], field=f"{field}.split"),
        ),
        fold_excess_log_growth=tuple(fold_values),
        mean_excess_log_growth=_number(
            payload["mean_excess_log_growth"],
            field=f"{field}.mean_excess_log_growth",
        ),
        lower_confidence_bound=_number(
            payload["lower_confidence_bound"],
            field=f"{field}.lower_confidence_bound",
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
            payload["test_evidence_digest"],
            field="stage_a_test.test_evidence_digest",
        ),
        selected_candidate_id=_string(
            payload["selected_candidate_id"],
            field="stage_a_test.selected_candidate_id",
        ),
        candidate_summary=_load_summary(
            payload["candidate_summary"], field="stage_a_test.candidate_summary"
        ),
        minimum_lower_bound=_number(
            payload["minimum_lower_bound"],
            field="stage_a_test.minimum_lower_bound",
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
        selection=selection,
        evidence=evidence,
    )
    if decision != expected:
        raise ValueError("Stage A sealed-test decision does not match recomputation")
    return decision


__all__ = [
    "STAGE_A_CANDIDATE_SUMMARY_SCHEMA",
    "STAGE_A_SEALED_TEST_DECISION_SCHEMA",
    "STAGE_A_VALIDATION_SELECTION_SCHEMA",
    "StageACandidateSummary",
    "StageASealedTestDecision",
    "StageAValidationSelection",
    "evaluate_stage_a_sealed_test",
    "load_stage_a_sealed_test_decision",
    "load_stage_a_validation_selection",
    "select_stage_a_validation_candidate",
    "summarize_stage_a_candidate",
    "write_stage_a_sealed_test_decision",
    "write_stage_a_validation_selection",
]
