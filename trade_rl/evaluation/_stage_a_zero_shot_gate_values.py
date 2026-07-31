"""Summary value objects for Stage A zero-shot evaluation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import StageAEvaluationSplit

STAGE_A_CANDIDATE_SUMMARY_SCHEMA: Final = "stage_a_zero_shot_candidate_summary_v2"
STAGE_A_VALIDATION_SELECTION_SCHEMA: Final = "stage_a_zero_shot_validation_selection_v2"
STAGE_A_SEALED_TEST_DECISION_SCHEMA: Final = "stage_a_zero_shot_sealed_test_decision_v2"
_RESAMPLING_UNIT: Final = "fold"
_TRIPLET_PASS_EXCESS_THRESHOLD: Final = 0.0
_BOOTSTRAP_CHUNK_SIZE: Final = 50_000


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _fraction(value: float, *, field: str) -> float:
    result = _finite(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return result


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
        raise ValueError(f"{field} must be boolean")
    return value


def _require_fields(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} field closure mismatch")


def _normalize_int_values(
    values: tuple[tuple[int, float], ...], *, field: str
) -> tuple[tuple[int, float], ...]:
    if len(values) < 2:
        raise ValueError(f"{field} requires at least two values")
    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for key, value in values:
        resolved = _non_negative_int(key, field=f"{field}.key")
        if resolved in seen:
            raise ValueError(f"{field} keys must be unique")
        seen.add(resolved)
        normalized.append((resolved, _finite(value, field=f"{field}.value")))
    return tuple(sorted(normalized))


def _normalize_digest_values(
    values: tuple[tuple[str, float], ...], *, field: str
) -> tuple[tuple[str, float], ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key, value in values:
        require_sha256(key, field=f"{field}.key")
        if key in seen:
            raise ValueError(f"{field} keys must be unique")
        seen.add(key)
        normalized.append((key, _finite(value, field=f"{field}.value")))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class StageACandidateSummary:
    """Fold bootstrap plus explicit unseen-triplet and seed robustness statistics."""

    plan_digest: str
    evidence_digest: str
    candidate_id: str
    split: StageAEvaluationSplit
    fold_excess_log_growth: tuple[tuple[int, float], ...]
    triplet_excess_log_growth: tuple[tuple[str, float], ...]
    seed_excess_log_growth: tuple[tuple[int, float], ...]
    mean_excess_log_growth: float
    lower_confidence_bound: float
    worst_triplet_excess_log_growth: float
    worst_seed_excess_log_growth: float
    triplet_pass_fraction: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    triplet_pass_excess_threshold: float = _TRIPLET_PASS_EXCESS_THRESHOLD
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
        fold_values = _normalize_int_values(
            self.fold_excess_log_growth, field="stage_a_summary.fold_excess_log_growth"
        )
        triplet_values = _normalize_digest_values(
            self.triplet_excess_log_growth,
            field="stage_a_summary.triplet_excess_log_growth",
        )
        seed_values = _normalize_int_values(
            self.seed_excess_log_growth, field="stage_a_summary.seed_excess_log_growth"
        )
        mean_value = _finite(
            self.mean_excess_log_growth, field="stage_a_summary.mean_excess_log_growth"
        )
        expected_mean = fmean(value for _, value in fold_values)
        if not math.isclose(mean_value, expected_mean, rel_tol=1e-15, abs_tol=1e-15):
            raise ValueError("Stage A candidate summary mean mismatch")
        lower_bound = _finite(
            self.lower_confidence_bound,
            field="stage_a_summary.lower_confidence_bound",
        )
        worst_triplet = _finite(
            self.worst_triplet_excess_log_growth,
            field="stage_a_summary.worst_triplet_excess_log_growth",
        )
        expected_worst_triplet = min(value for _, value in triplet_values)
        if not math.isclose(
            worst_triplet, expected_worst_triplet, rel_tol=1e-15, abs_tol=1e-15
        ):
            raise ValueError("Stage A candidate summary worst triplet mismatch")
        worst_seed = _finite(
            self.worst_seed_excess_log_growth,
            field="stage_a_summary.worst_seed_excess_log_growth",
        )
        expected_worst_seed = min(value for _, value in seed_values)
        if not math.isclose(
            worst_seed, expected_worst_seed, rel_tol=1e-15, abs_tol=1e-15
        ):
            raise ValueError("Stage A candidate summary worst seed mismatch")
        pass_threshold = _finite(
            self.triplet_pass_excess_threshold,
            field="stage_a_summary.triplet_pass_excess_threshold",
        )
        if pass_threshold != _TRIPLET_PASS_EXCESS_THRESHOLD:
            raise ValueError("Stage A candidate summary triplet pass threshold mismatch")
        pass_fraction = _fraction(
            self.triplet_pass_fraction, field="stage_a_summary.triplet_pass_fraction"
        )
        expected_pass_fraction = sum(
            value >= pass_threshold for _, value in triplet_values
        ) / len(triplet_values)
        if not math.isclose(
            pass_fraction, expected_pass_fraction, rel_tol=1e-15, abs_tol=1e-15
        ):
            raise ValueError("Stage A candidate summary triplet pass fraction mismatch")
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
        object.__setattr__(self, "triplet_excess_log_growth", triplet_values)
        object.__setattr__(self, "seed_excess_log_growth", seed_values)
        object.__setattr__(self, "mean_excess_log_growth", mean_value)
        object.__setattr__(self, "lower_confidence_bound", lower_bound)
        object.__setattr__(self, "worst_triplet_excess_log_growth", worst_triplet)
        object.__setattr__(self, "worst_seed_excess_log_growth", worst_seed)
        object.__setattr__(self, "triplet_pass_fraction", pass_fraction)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "triplet_pass_excess_threshold", pass_threshold)
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
            "seed_excess_log_growth": self.seed_excess_log_growth,
            "split": self.split,
            "triplet_excess_log_growth": self.triplet_excess_log_growth,
            "triplet_pass_excess_threshold": self.triplet_pass_excess_threshold,
            "triplet_pass_fraction": self.triplet_pass_fraction,
            "worst_seed_excess_log_growth": self.worst_seed_excess_log_growth,
            "worst_triplet_excess_log_growth": self.worst_triplet_excess_log_growth,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def _summary_meets_gate(
    summary: StageACandidateSummary,
    *,
    minimum_lower_bound: float,
    minimum_worst_triplet_excess: float,
    minimum_worst_seed_excess: float,
    minimum_triplet_pass_fraction: float,
) -> bool:
    return (
        summary.lower_confidence_bound >= minimum_lower_bound
        and summary.worst_triplet_excess_log_growth
        >= minimum_worst_triplet_excess
        and summary.worst_seed_excess_log_growth >= minimum_worst_seed_excess
        and summary.triplet_pass_fraction >= minimum_triplet_pass_fraction
    )


