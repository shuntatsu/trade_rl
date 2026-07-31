"""Immutable Stage A zero-shot evaluation plan."""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation._stage_a_zero_shot_candidate import StageACandidate
from trade_rl.evaluation._stage_a_zero_shot_contract_helpers import (
    MAX_STAGE_A_BOOTSTRAP_RESAMPLES,
    STAGE_A_EVALUATION_PLAN_SCHEMA,
    StageAEvaluationSplit,
    _finite,
    _fraction,
    _non_negative_int,
    _positive_int,
    _unique_digests,
    _unique_ints,
)


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
    minimum_validation_worst_triplet_excess: float
    minimum_test_worst_triplet_excess: float
    minimum_validation_worst_seed_excess: float
    minimum_test_worst_seed_excess: float
    minimum_validation_triplet_pass_fraction: float
    minimum_test_triplet_pass_fraction: float
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
            if tuple(seed for seed, _ in candidate.checkpoint_digests) != seeds:
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
            maximum=MAX_STAGE_A_BOOTSTRAP_RESAMPLES,
        )
        bootstrap_seed = _non_negative_int(
            self.bootstrap_seed, field="stage_a_evaluation_plan.bootstrap_seed"
        )
        finite_fields = {
            "minimum_validation_lower_bound": self.minimum_validation_lower_bound,
            "minimum_test_lower_bound": self.minimum_test_lower_bound,
            "minimum_validation_worst_triplet_excess": (
                self.minimum_validation_worst_triplet_excess
            ),
            "minimum_test_worst_triplet_excess": self.minimum_test_worst_triplet_excess,
            "minimum_validation_worst_seed_excess": (
                self.minimum_validation_worst_seed_excess
            ),
            "minimum_test_worst_seed_excess": self.minimum_test_worst_seed_excess,
        }
        normalized_finite = {
            field: _finite(value, field=f"stage_a_evaluation_plan.{field}")
            for field, value in finite_fields.items()
        }
        validation_pass_fraction = _fraction(
            self.minimum_validation_triplet_pass_fraction,
            field="stage_a_evaluation_plan.minimum_validation_triplet_pass_fraction",
        )
        test_pass_fraction = _fraction(
            self.minimum_test_triplet_pass_fraction,
            field="stage_a_evaluation_plan.minimum_test_triplet_pass_fraction",
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "validation_triplet_ids", validation_triplets)
        object.__setattr__(self, "test_triplet_ids", test_triplets)
        object.__setattr__(self, "bootstrap_confidence_level", confidence)
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        for field, numeric_value in normalized_finite.items():
            object.__setattr__(self, field, numeric_value)
        object.__setattr__(
            self, "minimum_validation_triplet_pass_fraction", validation_pass_fraction
        )
        object.__setattr__(
            self, "minimum_test_triplet_pass_fraction", test_pass_fraction
        )
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

    def triplet_ids_for(self, split: StageAEvaluationSplit | str) -> tuple[str, ...]:
        if split == "validation":
            return self.validation_triplet_ids
        if split == "test":
            return self.test_triplet_ids
        raise ValueError("Stage A evaluation split is invalid")

    def gate_thresholds(
        self, split: StageAEvaluationSplit | str
    ) -> tuple[float, float, float, float]:
        if split == "validation":
            return (
                self.minimum_validation_lower_bound,
                self.minimum_validation_worst_triplet_excess,
                self.minimum_validation_worst_seed_excess,
                self.minimum_validation_triplet_pass_fraction,
            )
        if split == "test":
            return (
                self.minimum_test_lower_bound,
                self.minimum_test_worst_triplet_excess,
                self.minimum_test_worst_seed_excess,
                self.minimum_test_triplet_pass_fraction,
            )
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
            "minimum_test_triplet_pass_fraction": self.minimum_test_triplet_pass_fraction,
            "minimum_test_worst_seed_excess": self.minimum_test_worst_seed_excess,
            "minimum_test_worst_triplet_excess": self.minimum_test_worst_triplet_excess,
            "minimum_validation_lower_bound": self.minimum_validation_lower_bound,
            "minimum_validation_triplet_pass_fraction": (
                self.minimum_validation_triplet_pass_fraction
            ),
            "minimum_validation_worst_seed_excess": (
                self.minimum_validation_worst_seed_excess
            ),
            "minimum_validation_worst_triplet_excess": (
                self.minimum_validation_worst_triplet_excess
            ),
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
