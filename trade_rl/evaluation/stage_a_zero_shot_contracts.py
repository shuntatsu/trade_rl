"""Immutable contracts for Stage A unseen-symbol evaluation evidence."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256

STAGE_A_CANDIDATE_SCHEMA: Final = "stage_a_zero_shot_candidate_v1"
STAGE_A_EVALUATION_PLAN_SCHEMA: Final = "stage_a_zero_shot_evaluation_plan_v1"
STAGE_A_OBSERVATION_SCHEMA: Final = "stage_a_zero_shot_observation_v1"
STAGE_A_EVIDENCE_SCHEMA: Final = "stage_a_zero_shot_evidence_v1"

StageAEvaluationSplit = Literal["validation", "test"]
_SPLITS: Final = frozenset({"validation", "test"})


def _non_negative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: int, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    return value


def _finite(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _unique_ints(
    values: tuple[int, ...], *, field: str, minimum_count: int = 1
) -> tuple[int, ...]:
    if len(values) < minimum_count:
        raise ValueError(f"{field} must contain at least {minimum_count} values")
    normalized = tuple(_non_negative_int(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _unique_strings(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized = tuple(require_non_empty(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(normalized))


def _unique_digests(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized = _unique_strings(values, field=field)
    for value in normalized:
        require_sha256(value, field=field)
    return normalized


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


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _require_fields(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} field closure mismatch")


@dataclass(frozen=True, slots=True)
class StageACandidate:
    """One exact trained candidate and its retained per-seed checkpoints."""

    candidate_id: str
    candidate_config_digest: str
    final_training_completion_digest: str
    policy_identity: str
    checkpoint_digests: tuple[tuple[int, str], ...]
    schema_version: str = STAGE_A_CANDIDATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_CANDIDATE_SCHEMA:
            raise ValueError("unsupported Stage A candidate schema")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_candidate.candidate_id"
        )
        for field, value in (
            ("candidate_config_digest", self.candidate_config_digest),
            (
                "final_training_completion_digest",
                self.final_training_completion_digest,
            ),
            ("policy_identity", self.policy_identity),
        ):
            require_sha256(value, field=f"stage_a_candidate.{field}")
        if not self.checkpoint_digests:
            raise ValueError("Stage A candidate checkpoints must not be empty")
        normalized: list[tuple[int, str]] = []
        seen: set[int] = set()
        for seed, digest in self.checkpoint_digests:
            resolved_seed = _non_negative_int(
                seed, field="stage_a_candidate.checkpoint_seed"
            )
            if resolved_seed in seen:
                raise ValueError("Stage A candidate checkpoint seeds must be unique")
            require_sha256(digest, field="stage_a_candidate.checkpoint_digest")
            seen.add(resolved_seed)
            normalized.append((resolved_seed, digest))
        normalized_checkpoints = tuple(sorted(normalized))
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "checkpoint_digests", normalized_checkpoints)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A candidate digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        candidate_config_digest: str,
        final_training_completion_digest: str,
        policy_identity: str,
        checkpoint_digests: tuple[tuple[int, str], ...],
    ) -> StageACandidate:
        return cls(
            candidate_id=candidate_id,
            candidate_config_digest=candidate_config_digest,
            final_training_completion_digest=final_training_completion_digest,
            policy_identity=policy_identity,
            checkpoint_digests=checkpoint_digests,
        )

    def checkpoint_digest(self, seed: int) -> str:
        resolved = _non_negative_int(seed, field="stage_a_candidate.seed")
        try:
            return dict(self.checkpoint_digests)[resolved]
        except KeyError as error:
            raise ValueError("Stage A candidate checkpoint seed is not declared") from error

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "candidate_id": self.candidate_id,
            "checkpoint_digests": self.checkpoint_digests,
            "final_training_completion_digest": self.final_training_completion_digest,
            "policy_identity": self.policy_identity,
            "schema_version": self.schema_version,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageAZeroShotEvaluationPlan:
    """Predeclared identities and statistical thresholds for Stage A evaluation."""

    symbol_disjoint_manifest_digest: str
    symbol_disjoint_triplet_manifest_digest: str
    dataset_identity: str
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    candidates: tuple[StageACandidate, ...]
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    validation_triplet_ids: tuple[str, ...]
    test_triplet_ids: tuple[str, ...]
    bootstrap_confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    minimum_validation_lower_bound: float
    minimum_test_lower_bound: float
    schema_version: str = STAGE_A_EVALUATION_PLAN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EVALUATION_PLAN_SCHEMA:
            raise ValueError("unsupported Stage A evaluation plan schema")
        for field, value in (
            ("symbol_disjoint_manifest_digest", self.symbol_disjoint_manifest_digest),
            (
                "symbol_disjoint_triplet_manifest_digest",
                self.symbol_disjoint_triplet_manifest_digest,
            ),
            ("dataset_identity", self.dataset_identity),
            ("feature_identity", self.feature_identity),
            ("execution_identity", self.execution_identity),
            ("evaluation_identity", self.evaluation_identity),
        ):
            require_sha256(value, field=f"stage_a_evaluation_plan.{field}")
        if not self.candidates:
            raise ValueError("Stage A evaluation candidates must not be empty")
        candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Stage A evaluation candidate IDs must be unique")
        seeds = _unique_ints(
            tuple(self.seeds), field="stage_a_evaluation_plan.seeds", minimum_count=2
        )
        folds = _unique_ints(
            tuple(self.folds), field="stage_a_evaluation_plan.folds", minimum_count=2
        )
        for candidate in candidates:
            checkpoint_seeds = tuple(seed for seed, _ in candidate.checkpoint_digests)
            if checkpoint_seeds != seeds:
                raise ValueError("Stage A candidate checkpoint seed closure mismatch")
        validation_triplets = _unique_digests(
            tuple(self.validation_triplet_ids),
            field="stage_a_evaluation_plan.validation_triplet_ids",
        )
        test_triplets = _unique_digests(
            tuple(self.test_triplet_ids),
            field="stage_a_evaluation_plan.test_triplet_ids",
        )
        if not set(validation_triplets).isdisjoint(test_triplets):
            raise ValueError("Stage A validation and test triplets must be disjoint")
        confidence = _finite(
            self.bootstrap_confidence_level,
            field="stage_a_evaluation_plan.bootstrap_confidence_level",
        )
        if not 0.5 < confidence < 1.0:
            raise ValueError("Stage A bootstrap confidence must be within (0.5, 1)")
        resamples = _positive_int(
            self.bootstrap_resamples,
            field="stage_a_evaluation_plan.bootstrap_resamples",
            minimum=1_000,
        )
        bootstrap_seed = _non_negative_int(
            self.bootstrap_seed, field="stage_a_evaluation_plan.bootstrap_seed"
        )
        validation_threshold = _finite(
            self.minimum_validation_lower_bound,
            field="stage_a_evaluation_plan.minimum_validation_lower_bound",
        )
        test_threshold = _finite(
            self.minimum_test_lower_bound,
            field="stage_a_evaluation_plan.minimum_test_lower_bound",
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "validation_triplet_ids", validation_triplets)
        object.__setattr__(self, "test_triplet_ids", test_triplets)
        object.__setattr__(self, "bootstrap_confidence_level", confidence)
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "minimum_validation_lower_bound", validation_threshold)
        object.__setattr__(self, "minimum_test_lower_bound", test_threshold)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A evaluation plan digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    def candidate(self, candidate_id: str) -> StageACandidate:
        resolved = require_non_empty(candidate_id, field="stage_a_candidate_id")
        for candidate in self.candidates:
            if candidate.candidate_id == resolved:
                return candidate
        raise ValueError("Stage A evaluation candidate is not declared")

    def triplet_ids_for(self, split: StageAEvaluationSplit) -> tuple[str, ...]:
        if split == "validation":
            return self.validation_triplet_ids
        if split == "test":
            return self.test_triplet_ids
        raise ValueError("Stage A evaluation split is invalid")

    def digest_payload(self) -> dict[str, object]:
        return {
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "candidates": tuple(item.to_json_dict() for item in self.candidates),
            "dataset_identity": self.dataset_identity,
            "evaluation_identity": self.evaluation_identity,
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "folds": self.folds,
            "minimum_test_lower_bound": self.minimum_test_lower_bound,
            "minimum_validation_lower_bound": self.minimum_validation_lower_bound,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "symbol_disjoint_manifest_digest": self.symbol_disjoint_manifest_digest,
            "symbol_disjoint_triplet_manifest_digest": (
                self.symbol_disjoint_triplet_manifest_digest
            ),
            "test_triplet_ids": self.test_triplet_ids,
            "validation_triplet_ids": self.validation_triplet_ids,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageAEvaluationObservation:
    """Paired candidate/baseline growth for one exact evaluation cell."""

    candidate_id: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    checkpoint_digest: str
    dataset_identity: str
    execution_evidence_digest: str
    policy_log_growth: float
    baseline_log_growth: float
    schema_version: str = STAGE_A_OBSERVATION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_OBSERVATION_SCHEMA:
            raise ValueError("unsupported Stage A observation schema")
        candidate_id = require_non_empty(
            self.candidate_id, field="stage_a_observation.candidate_id"
        )
        if self.split not in _SPLITS:
            raise ValueError("Stage A observation split is invalid")
        for field, value in (
            ("triplet_id", self.triplet_id),
            ("checkpoint_digest", self.checkpoint_digest),
            ("dataset_identity", self.dataset_identity),
            ("execution_evidence_digest", self.execution_evidence_digest),
        ):
            require_sha256(value, field=f"stage_a_observation.{field}")
        fold = _non_negative_int(self.fold, field="stage_a_observation.fold")
        seed = _non_negative_int(self.seed, field="stage_a_observation.seed")
        policy_growth = _finite(
            self.policy_log_growth, field="stage_a_observation.policy_log_growth"
        )
        baseline_growth = _finite(
            self.baseline_log_growth,
            field="stage_a_observation.baseline_log_growth",
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "policy_log_growth", policy_growth)
        object.__setattr__(self, "baseline_log_growth", baseline_growth)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A observation digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        split: str,
        triplet_id: str,
        fold: int,
        seed: int,
        checkpoint_digest: str,
        dataset_identity: str,
        execution_evidence_digest: str,
        policy_log_growth: float,
        baseline_log_growth: float,
    ) -> StageAEvaluationObservation:
        return cls(
            candidate_id=candidate_id,
            split=cast(StageAEvaluationSplit, split),
            triplet_id=triplet_id,
            fold=fold,
            seed=seed,
            checkpoint_digest=checkpoint_digest,
            dataset_identity=dataset_identity,
            execution_evidence_digest=execution_evidence_digest,
            policy_log_growth=policy_log_growth,
            baseline_log_growth=baseline_log_growth,
        )

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.candidate_id, self.triplet_id, self.fold, self.seed)

    @property
    def excess_log_growth(self) -> float:
        return self.policy_log_growth - self.baseline_log_growth

    def digest_payload(self) -> dict[str, object]:
        return {
            "baseline_log_growth": self.baseline_log_growth,
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_identity": self.dataset_identity,
            "execution_evidence_digest": self.execution_evidence_digest,
            "fold": self.fold,
            "policy_log_growth": self.policy_log_growth,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split": self.split,
            "triplet_id": self.triplet_id,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


@dataclass(frozen=True, slots=True)
class StageAEvaluationEvidence:
    """Complete Cartesian evidence for one split and candidate set."""

    plan_digest: str
    split: StageAEvaluationSplit
    candidate_ids: tuple[str, ...]
    folds: tuple[int, ...]
    seeds: tuple[int, ...]
    triplet_ids: tuple[str, ...]
    observations: tuple[StageAEvaluationObservation, ...]
    schema_version: str = STAGE_A_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EVIDENCE_SCHEMA:
            raise ValueError("unsupported Stage A evidence schema")
        require_sha256(self.plan_digest, field="stage_a_evidence.plan_digest")
        if self.split not in _SPLITS:
            raise ValueError("Stage A evidence split is invalid")
        candidate_ids = _unique_strings(
            tuple(self.candidate_ids), field="stage_a_evidence.candidate_ids"
        )
        folds = _unique_ints(
            tuple(self.folds), field="stage_a_evidence.folds", minimum_count=2
        )
        seeds = _unique_ints(
            tuple(self.seeds), field="stage_a_evidence.seeds", minimum_count=2
        )
        triplet_ids = _unique_digests(
            tuple(self.triplet_ids), field="stage_a_evidence.triplet_ids"
        )
        observations = tuple(sorted(self.observations, key=lambda item: item.key))
        keys = tuple(item.key for item in observations)
        if len(set(keys)) != len(keys):
            raise ValueError("Stage A evidence contains a duplicate observation")
        expected_keys = set(
            itertools.product(candidate_ids, triplet_ids, folds, seeds)
        )
        if set(keys) != expected_keys:
            raise ValueError("Stage A evidence observation closure mismatch")
        if any(item.split != self.split for item in observations):
            raise ValueError("Stage A evidence contains a cross-split observation")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "triplet_ids", triplet_ids)
        object.__setattr__(self, "observations", observations)
        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("Stage A evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def validate_plan(self, plan: StageAZeroShotEvaluationPlan) -> None:
        if self.plan_digest != plan.digest:
            raise ValueError("Stage A evidence plan digest mismatch")
        if self.folds != plan.folds or self.seeds != plan.seeds:
            raise ValueError("Stage A evidence fold or seed closure mismatch")
        if self.triplet_ids != plan.triplet_ids_for(self.split):
            raise ValueError("Stage A evidence triplet closure mismatch")
        if not set(self.candidate_ids) <= set(plan.candidate_ids):
            raise ValueError("Stage A evidence contains an undeclared candidate")
        for observation in self.observations:
            candidate = plan.candidate(observation.candidate_id)
            if observation.dataset_identity != plan.dataset_identity:
                raise ValueError("Stage A observation dataset identity mismatch")
            if observation.checkpoint_digest != candidate.checkpoint_digest(
                observation.seed
            ):
                raise ValueError("Stage A observation checkpoint digest mismatch")

    def observations_for(self, candidate_id: str) -> tuple[StageAEvaluationObservation, ...]:
        resolved = require_non_empty(candidate_id, field="stage_a_candidate_id")
        if resolved not in self.candidate_ids:
            raise ValueError("Stage A evidence candidate is not present")
        return tuple(
            item for item in self.observations if item.candidate_id == resolved
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_ids": self.candidate_ids,
            "folds": self.folds,
            "observations": tuple(item.to_json_dict() for item in self.observations),
            "plan_digest": self.plan_digest,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "split": self.split,
            "triplet_ids": self.triplet_ids,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_stage_a_zero_shot_evaluation_plan(
    *,
    symbol_disjoint_manifest_digest: str,
    symbol_disjoint_triplet_manifest_digest: str,
    dataset_identity: str,
    feature_identity: str,
    execution_identity: str,
    evaluation_identity: str,
    candidates: tuple[StageACandidate, ...],
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    validation_triplet_ids: tuple[str, ...],
    test_triplet_ids: tuple[str, ...],
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    minimum_validation_lower_bound: float,
    minimum_test_lower_bound: float,
) -> StageAZeroShotEvaluationPlan:
    return StageAZeroShotEvaluationPlan(
        symbol_disjoint_manifest_digest=symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=(
            symbol_disjoint_triplet_manifest_digest
        ),
        dataset_identity=dataset_identity,
        feature_identity=feature_identity,
        execution_identity=execution_identity,
        evaluation_identity=evaluation_identity,
        candidates=candidates,
        seeds=seeds,
        folds=folds,
        validation_triplet_ids=validation_triplet_ids,
        test_triplet_ids=test_triplet_ids,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        minimum_validation_lower_bound=minimum_validation_lower_bound,
        minimum_test_lower_bound=minimum_test_lower_bound,
    )


def build_stage_a_evaluation_evidence(
    *,
    plan: StageAZeroShotEvaluationPlan,
    split: str,
    observations: tuple[StageAEvaluationObservation, ...],
    candidate_ids: tuple[str, ...] | None = None,
) -> StageAEvaluationEvidence:
    resolved_split = cast(StageAEvaluationSplit, split)
    selected_candidates = plan.candidate_ids if candidate_ids is None else candidate_ids
    evidence = StageAEvaluationEvidence(
        plan_digest=plan.digest,
        split=resolved_split,
        candidate_ids=selected_candidates,
        folds=plan.folds,
        seeds=plan.seeds,
        triplet_ids=plan.triplet_ids_for(resolved_split),
        observations=observations,
    )
    evidence.validate_plan(plan)
    return evidence


def write_stage_a_zero_shot_evaluation_plan(
    path: str | Path, plan: StageAZeroShotEvaluationPlan
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(plan.to_json_dict()))
    return output


def write_stage_a_evaluation_evidence(
    path: str | Path, evidence: StageAEvaluationEvidence
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(evidence.to_json_dict()))
    return output


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
    raw_checkpoints = _list(payload["checkpoint_digests"], field=f"{field}.checkpoint_digests")
    checkpoints: list[tuple[int, str]] = []
    for index, raw in enumerate(raw_checkpoints):
        pair = _list(raw, field=f"{field}.checkpoint_digests[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{field}.checkpoint_digests[{index}] must contain two values")
        checkpoints.append(
            (
                _integer(pair[0], field=f"{field}.checkpoint_digests[{index}].seed"),
                _string(pair[1], field=f"{field}.checkpoint_digests[{index}].digest"),
            )
        )
    return StageACandidate(
        candidate_id=_string(payload["candidate_id"], field=f"{field}.candidate_id"),
        candidate_config_digest=_string(
            payload["candidate_config_digest"],
            field=f"{field}.candidate_config_digest",
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
            "minimum_validation_lower_bound",
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
            _integer(value, field="stage_a_plan.seeds")
            for value in _list(payload["seeds"], field="stage_a_plan.seeds")
        ),
        folds=tuple(
            _integer(value, field="stage_a_plan.folds")
            for value in _list(payload["folds"], field="stage_a_plan.folds")
        ),
        validation_triplet_ids=tuple(
            _string(value, field="stage_a_plan.validation_triplet_ids")
            for value in _list(
                payload["validation_triplet_ids"],
                field="stage_a_plan.validation_triplet_ids",
            )
        ),
        test_triplet_ids=tuple(
            _string(value, field="stage_a_plan.test_triplet_ids")
            for value in _list(
                payload["test_triplet_ids"], field="stage_a_plan.test_triplet_ids"
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
            "baseline_log_growth",
            "candidate_id",
            "checkpoint_digest",
            "dataset_identity",
            "digest",
            "execution_evidence_digest",
            "fold",
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
            StageAEvaluationSplit,
            _string(payload["split"], field=f"{field}.split"),
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
        execution_evidence_digest=_string(
            payload["execution_evidence_digest"],
            field=f"{field}.execution_evidence_digest",
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
            _string(value, field="stage_a_evidence.candidate_ids")
            for value in _list(
                payload["candidate_ids"], field="stage_a_evidence.candidate_ids"
            )
        ),
        folds=tuple(
            _integer(value, field="stage_a_evidence.folds")
            for value in _list(payload["folds"], field="stage_a_evidence.folds")
        ),
        seeds=tuple(
            _integer(value, field="stage_a_evidence.seeds")
            for value in _list(payload["seeds"], field="stage_a_evidence.seeds")
        ),
        triplet_ids=tuple(
            _string(value, field="stage_a_evidence.triplet_ids")
            for value in _list(
                payload["triplet_ids"], field="stage_a_evidence.triplet_ids"
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


__all__ = [
    "STAGE_A_CANDIDATE_SCHEMA",
    "STAGE_A_EVALUATION_PLAN_SCHEMA",
    "STAGE_A_EVIDENCE_SCHEMA",
    "STAGE_A_OBSERVATION_SCHEMA",
    "StageACandidate",
    "StageAEvaluationEvidence",
    "StageAEvaluationObservation",
    "StageAEvaluationSplit",
    "StageAZeroShotEvaluationPlan",
    "build_stage_a_evaluation_evidence",
    "build_stage_a_zero_shot_evaluation_plan",
    "load_stage_a_evaluation_evidence",
    "load_stage_a_zero_shot_evaluation_plan",
    "write_stage_a_evaluation_evidence",
    "write_stage_a_zero_shot_evaluation_plan",
]
